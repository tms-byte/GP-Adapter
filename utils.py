from tqdm import tqdm

import glob
import os

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from yacs.config import CfgNode as CN


_tokenizer = _Tokenizer()


def _to_cfg_node(value):
    if isinstance(value, dict):
        node = CN(new_allowed=True)
        for key, child in value.items():
            node[key] = _to_cfg_node(child)
        return node
    if isinstance(value, list):
        return [_to_cfg_node(child) for child in value]
    return value



def _build_prompt_cfg(data):
    cfg = CN(new_allowed=True)
    cfg.DATALOADER = CN(new_allowed=True)
    cfg.INPUT = CN(new_allowed=True)
    cfg.INPUT.SIZE = (224, 224)
    cfg.TRAIN = CN(new_allowed=True)
    cfg.OPTIM = CN(new_allowed=True)

    cfg.TRAINER = CN(new_allowed=True)
    cfg.TRAINER.LOCOOP = CN(new_allowed=True)
    cfg.TRAINER.LOCOOP.N_CTX = 16
    cfg.TRAINER.LOCOOP.CTX_INIT = ""
    cfg.TRAINER.LOCOOP.CSC = False
    cfg.TRAINER.LOCOOP.CLASS_TOKEN_POSITION = "end"

    cfg.MODEL = CN(new_allowed=True)
    cfg.MODEL.BACKBONE = CN(new_allowed=True)
    cfg.MODEL.BACKBONE.NAME = "ViT-B/16"

    for key, value in (data or {}).items():
        cfg[key] = _to_cfg_node(value)

    if isinstance(cfg.INPUT.SIZE, str):
        size_text = cfg.INPUT.SIZE.strip().strip("()")
        if size_text:
            cfg.INPUT.SIZE = tuple(int(part.strip()) for part in size_text.split(",") if part.strip())
    elif isinstance(cfg.INPUT.SIZE, list):
        cfg.INPUT.SIZE = tuple(cfg.INPUT.SIZE)

    return cfg


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        # 各クラスラベルに対する重みを生成する。これは特徴量数の次元384だっけ？を持つ
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # print(self.transformer(x))
        # x, _, _, _ = self.transformer(x)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.LOCOOP.N_CTX
        ctx_init = cfg.TRAINER.LOCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.LOCOOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        # tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        # with torch.no_grad():
        #     embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        # (修正後) ーーー GPUへ移動＆catをやめてバッチ化 ーーー
        # ※ catで1件ずつtokenizeするより、まとめてtokenizeした方が安全＆高速です
        tokenized_prompts = clip.tokenize(prompts).to(clip_model.token_embedding.weight.device)  # ← ここがポイント！
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)


        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.LOCOOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


def clip_classifier_locoop(
    classnames,
    clip_model,
    num_samples: int = 16,       # クラスごとのサンプル数
    sigma: float = 0.02,         # ノイズ強度
    use_gaussian_sampling: bool = False,  # FalseでLoCoOp埋め込み1本のみ返す
    # --- ミキシング（テンプレ平均とLoCoOpの線形ブレンド） ---
    alpha: float = None,         # Noneか未指定: LoCoOpのみ。0〜1なら(1-alpha)*template + alpha*LoCoOp
    template: list = None,       # 例: ["a photo of a {}.", ...]。alpha指定時のみ使用
    device: str = "cuda",
    prompt_ckpt_path : str = None,
    cfg_path : str = None,
    # prompt_ckpt_path : str = "/home/saito/docker/Tip-Adapter/locoop/vit_b16_ep50_16shots/nctx16_cscFalse_ctpend/seed1/",
    cfg = None
):
    """
    戻り値:
        use_gaussian_sampling=True  : [num_classes, num_samples, feat_dim]
        use_gaussian_sampling=False : [num_classes, feat_dim]
        依存:
        - LoCoOpのコードベース (PromptLearner/TextEncoder/load_clip_to_cpu, clip_w_local.clip)。
    期待:
        - cfg.MODEL.BACKBONE.NAME と cfg.INPUT.SIZE[0] が LoCoOp実装のアサート条件を満たすこと。
        - prompt_ckpt_path は `.../prompt_learner/model-best.pth.tar`（または同等）を指すか、
          その親ディレクトリ（関数内で推測）を指す。
    """

    if prompt_ckpt_path is None:
        raise ValueError("prompt_ckpt_path is required for LoCoOp/CoOp text features.")
    if cfg_path is None:
        raise ValueError("cfg_path is required for LoCoOp/CoOp text features.")

    data = yaml.load(open(cfg_path, 'r'), Loader=yaml.Loader)
    cfg = _build_prompt_cfg(data)
    # 2) PromptLearner 構築 & 学習済み重みロード
    prompt_learner = PromptLearner(cfg, classnames, clip_model).to(device).eval()

    ckpt_file = prompt_ckpt_path
    if os.path.isdir(prompt_ckpt_path):
        prompt_learner_dir = os.path.join(prompt_ckpt_path, "prompt_learner")
        search_roots = []
        if os.path.isdir(prompt_learner_dir):
            search_roots.append(prompt_learner_dir)
        search_roots.append(prompt_ckpt_path)

        # 典型的なファイル名たちを優先的に探索
        candidate_names = [
            "model-best.pth.tar",
            "model_best.pth.tar",
            "model.pth.tar",
            "model.pt",
        ]
        for root in search_roots:
            for name in candidate_names:
                candidate = os.path.join(root, name)
                if os.path.isfile(candidate):
                    ckpt_file = candidate
                    break
            if ckpt_file != prompt_ckpt_path:
                break

        # LoCoOpのcheckpointファイルが指す最新エポックを参照
        if ckpt_file == prompt_ckpt_path and os.path.isdir(prompt_learner_dir):
            ckpt_indicator = os.path.join(prompt_learner_dir, "checkpoint")
            if os.path.isfile(ckpt_indicator):
                with open(ckpt_indicator, "r") as f:
                    rel_path = f.readline().strip()
                candidate = os.path.join(prompt_learner_dir, rel_path)
                if rel_path and os.path.isfile(candidate):
                    ckpt_file = candidate

        # それでも見つからなければ *.pth* / *.pt を総当たりして最後に更新されたものを採用
        if ckpt_file == prompt_ckpt_path:
            globbed = []
            for root in search_roots:
                globbed.extend(glob.glob(os.path.join(root, "*.pth*")))
                globbed.extend(glob.glob(os.path.join(root, "*.pt")))
            globbed = [path for path in globbed if os.path.isfile(path)]
            if globbed:
                globbed.sort(key=os.path.getmtime, reverse=True)
                ckpt_file = globbed[0]
            else:
                raise FileNotFoundError(f"LoCoOp checkpoint not found under {prompt_ckpt_path}")

    ckpt = torch.load(ckpt_file, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    # token_prefix/suffixはロード時に無視してOK（LoCoOpのTrainer実装も同様の扱い）
    state = {k: v for k, v in state.items() if ("token_prefix" not in k and "token_suffix" not in k)}
    prompt_learner.load_state_dict(state, strict=False)

    # 3) TextEncoder 構築
    text_encoder = TextEncoder(clip_model).to(device).eval()

    # 4) LoCoOp学習済みのプロンプト列 → テキスト埋め込み（クラスごと1本）
    prompts = prompt_learner()  # [n_cls, n_tokens, dim] の連続埋め込み
    tokenized_prompts = prompt_learner.tokenized_prompts.to(device)
    text_feats_locoop = text_encoder(prompts, tokenized_prompts)       # [n_cls, d]
    text_feats_locoop = F.normalize(text_feats_locoop, dim=-1).float() # 正規化 & float32

    n_cls, feat_dim = text_feats_locoop.shape
    weights = []

    # 5) （必要なら）テンプレ・アンサンブルとのミキシング準備
    use_mix = (alpha is not None) and (template is not None)
    if use_mix:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0,1] when template is provided.")

    # 6) クラスごとに中心ベクトルを決め、必要ならノイズ注入でサンプル生成
    for i, cname in enumerate(classnames):
        mu_locoop = text_feats_locoop[i]  # [d]

        if use_mix:
            # テンプレート・アンサンブルの平均ベクトル
            cname_txt = cname.replace("_", " ")
            texts = [t.format(cname_txt) for t in template]
            tokens = clip.tokenize(texts).to(device)
            embeds = clip_model.encode_text(tokens)                 # [num_templates, d]
            embeds = F.normalize(embeds, dim=-1)
            mu_tmpl = embeds.mean(dim=0).float()                    # [d]

            # 線形ブレンド → 正規化
            mu = F.normalize(alpha * mu_locoop + (1.0 - alpha) * mu_tmpl, dim=-1)
        else:
            mu = mu_locoop

        if use_gaussian_sampling:
            # ガウスノイズ注入（正規化済み単位球上の近傍をサンプリング）
            eps = torch.randn(num_samples, feat_dim, device=device, dtype=mu.dtype)
            samples = mu.unsqueeze(0) + sigma * eps                 # [num_samples, d]
            samples = F.normalize(samples, dim=-1)
            weights.append(samples)
        else:
            weights.append(mu)

    if use_gaussian_sampling:
        clip_weights = torch.stack(weights, dim=0).contiguous()     # [n_cls, num_samples, d]
    else:
        clip_weights = torch.stack(weights, dim=0).contiguous()     # [n_cls, d]
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache):
    """
    キャッシュモデルを作成する関数。
    つまり、keyを作成する。全プロトタイプデータをCLIPの視覚特徴量変換する。
    """
    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    
    return features, labels
