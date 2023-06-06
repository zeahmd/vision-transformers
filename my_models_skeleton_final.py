import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# ViT
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# ViT & CrossViT
# PreNorm class for layer normalization before passing the input to another function (fn)
class PreNorm(nn.Module):    
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# ViT & CrossViT
# FeedForward class for a feed forward neural network consisting of 2 linear layers, 
# where the first has a GELU activation followed by dropout, then the next linear layer
# followed again by dropout
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        """FeedForward: nn.Module

        Parameters
        ----------
        dim : int
            embedding dimension (64, 256 or 512)
        hidden_dim : int
            number of units in the hidden layer of the feed forward network
        dropout : float
            dropout rate used with in the pointwise feed forward network
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# ViT & CrossViT
# Attention class for multi-head self-attention mechanism with softmax and dropout
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        """Attention: nn.Module
        
        Parameters
        ----------
        dim : int
            embedding dimension (64, 256 or 512)
        heads : int
            number of heads in the multi-head self-attention
        dim_head : int
            size of the head in multi-head self-attention (dim_head = dim / heads)
        dropout : float
            dropout rate applied to the output of attention heads
        """
        super().__init__()
        # set heads and scale (=sqrt(dim_head))
        self.heads = heads
        self.scale = dim_head ** -0.5
        # we need softmax layer and dropout
        self.softmax = nn.Softmax(dim=-1) # it will be applied to the attention scores which means that
        # for a patch k, in the attention head j, in the image i, the sum of the attention scores with all other patches
        # will be 1. This is because the attention scores are probabilities and the sum of probabilities is 1.
        # as well as the q linear layer
        self.q = nn.Linear(dim, dim_head*heads) # TODO: shouldn't it be dim_head*heads instead of dim_head?
        # and the k/v linear layer (can be realized as one single linear layer
        # or as two individual ones)
        self.k = nn.Linear(dim, dim_head*heads)
        self.v = nn.Linear(dim, dim_head*heads)
        # and the output linear layer followed by dropout
        self.out = nn.Linear(dim_head*heads, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None, kv_include_self = False):
        # now compute the attention/cross-attention
        # in cross attention: x = class token, context = token embeddings
        # don't forget the dropout after the attention 
        # and before the multiplication w. 'v'
        # the output should be in the shape 'b n (h d)'
        # x: (num_imgs, num_patches + 1, dim)


        # if context is None, then it is self-attention because context = x, otherwise it is cross-attention
        # in cross attention, context = concat(cls_token, second_tansformer_patch_tokens), and query = 1st_transformer_cls_token
        # in self-attention, context = x, and query = x. "context includes key and value both."
        # we actually query the context(key, value)

        b, n, _, h = *x.shape, self.heads
        if context is None:
            context = x

        if kv_include_self:
            # cross attention requires CLS token includes itself as key / value
            context = torch.cat((x, context), dim=1) 
        
        bq = self.q(x) # bq: (num_imgs, num_patches + 1, dim_head*heads)
        bk = self.k(context) # bk: (num_imgs, num_patches + 1, dim_head*heads)
        bv = self.v(context) # bv: (num_imgs, num_patches + 1, dim_head*heads)

        # split heads
        bq = rearrange(bq, 'b n (h d) -> b h n d', h=h) # bq: (num_imgs, heads, num_patches + 1, dim_head)
        bk = rearrange(bk, 'b n (h d) -> b h n d', h=h) # bk: (num_imgs, heads, num_patches + 1, dim_head)
        bv = rearrange(bv, 'b n (h d) -> b h n d', h=h) # bv: (num_imgs, heads, num_patches + 1, dim_head)

        # compute the attention scores
        # replace 'b h n d, b h d n1 -> b h n n1' with 'b h n d, b h d m -> b h n m' to make einsum work and
        # get the cross attention scores
        # attention_score: (num_imgs, heads, num_patches + 1, num_patches + 1)
        # we compute the attention score for each patch in the image with all other patches in the image
        attention_score = einsum('b h n d, b h d m -> b h n m', bq, rearrange(bk, 'b h n d -> b h d n')) * self.scale
        # apply softmax
        attention_score = self.softmax(attention_score) # attention_score can be visualized as a heatmap
        # apply dropout
        attention_score_dot_v = self.dropout(attention_score) 
        # multiply w. value
        attention_score_dot_v = einsum('b h n m, b h m d -> b h n d', attention_score_dot_v, bv) # attention_score_dot_v: (num_imgs, heads, num_patches + 1, dim_head)

        # merge heads
        attention_score_dot_v_heads_merged = rearrange(attention_score_dot_v, 'b h n d -> b n (h d)') # attention_score_heads_merged: (num_imgs, num_patches + 1, dim_head*heads)
        out = self.out(attention_score_dot_v_heads_merged)

        return out 


# ViT & CrossViT
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        """Transformer: nn.Module
        
        Parameters
        ----------
        dim : int
            embedding dimension (64, 256 or 512)
        depth : int
            defines the number of transformer blocks stacked on top of each other
        heads : int
            number of heads in the multi-head self-attention
        dim_head : int
            size of the head in multi-head self-attention (dim_head = dim / heads)
        mlp_dim : int
            number of units in the feed forward network in transformer blocks
        dropout : float
            dropout rate applied to the ouput of attention heads and the output of the feed forward network
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# CrossViT
# projecting CLS tokens, in the case that small and large patch tokens have different dimensions
class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        """ProjectInOut: It makes sure that the embedding size of small and large transformer blocks are the same by projecting dim_out to dim_in if necessary.

        Parameters
        ----------
        dim_in : int
            size of the input embedding
        dim_out : int
            size of the output embedding
        fn : string
            specifies whether to use function 'f' or 'g' which is 'projection' and 'back-projection' respectively
        """
        super().__init__()
        self.fn = fn
        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        if self.fn == 'f':
            x = self.project_in(x)
        elif self.fn == 'g':
            x = self.project_out(x)
        return x

# CrossViT
# cross attention transformer module
class CrossTransformer(nn.Module):
    # This is a special transformer block
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        # TODO: create # depth encoders using ProjectInOut
        # self.project_in_out = ProjectInOut(sm_dim, lg_dim, 'f')
        # Note: no positional FFN here


        # NOTE: Here I have implemented ProjectInOut with f and g functions, however in the
        # implementation of "Prof. Vincent" he didn't use 'f' and 'g' functions but passed
        # Attention(...) to ProjectInOut which first projects-in then perform attention, then
        # projects-out. Maybe that's bit compact implementation but more error prone.
        # Similar thing is done in "PreNorm" class where the input is first normalized then
        # passed to the function (fn) and then output is returned.
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(sm_dim, lg_dim, 'f'), # forward projection 'f' from small to large
                PreNorm(lg_dim, Attention(lg_dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                ProjectInOut(sm_dim, lg_dim, 'g'), # backward projection 'g' from large to small but stil it is (sm_dim, lg_dim) because g defines the projection from large to small
                ProjectInOut(lg_dim, sm_dim, 'f'),
                PreNorm(sm_dim, Attention(sm_dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                ProjectInOut(lg_dim, sm_dim, 'g')
            ]))

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        # TODO: make sure it works
        # Forward pass through the layers, 
        # cross attend to 
        # 1. small cls token to large patches and
        # 2. large cls token to small patches
        for project_in_sm, attn_sm, project_out_sm, project_in_lg, attn_lg, project_out_lg in self.layers:
            sm_cls =  project_in_sm(sm_cls)
            sm_cls = attn_sm(sm_cls, context=lg_patch_tokens, kv_include_self=True) + sm_cls
            sm_cls = project_out_sm(sm_cls)

            lg_cls = project_in_lg(lg_cls)
            lg_cls = attn_lg(lg_cls, context=sm_patch_tokens, kv_include_self=True) + lg_cls
            lg_cls = project_out_lg(lg_cls)

        # finally concat sm/lg cls tokens with patch tokens 
        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim=1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim=1)

        return sm_tokens, lg_tokens

# CrossViT
# multi-scale encoder
class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        """MultiScaleEncoder: combines small and large patch transformers with cross attention transformer
        
        Parameters
        ----------
        depth : int
            defines the number of transformer blocks stacked on top of each other
        sm_dim : int
            size of the small patch embedding in small patch transformer
        lg_dim : int
            size of the large patch embedding in large patch transformer
        sm_enc_params : dict
            parameters for small patch transformer
        lg_enc_params : dict
            parameters for large patch transformer
        cross_attn_heads : int
            number of heads in the cross attention transformer
        cross_attn_depth : int
            defines the number of cross attention transformer blocks stacked on top of each other
        cross_attn_dim_head : int
            size of the attention head in cross attention transformer (cross_attn_dim_head = dim / cross_attn_heads)
        dropout : float
            dropout rate applied to the output of the transformer blocks
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # 2 transformer branches, one for small, one for large patchs
                Transformer(sm_dim, **sm_enc_params),
                Transformer(lg_dim, **lg_enc_params),
                # + 1 cross transformer 
                CrossTransformer(sm_dim, lg_dim, cross_attn_depth, cross_attn_heads, cross_attn_dim_head, dropout)
            ]))

    def forward(self, sm_tokens, lg_tokens):
        # forward through the transformer encoders and cross attention block
        for sm_enc, lg_enc, cross_attn in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attn(sm_tokens, lg_tokens)
        return sm_tokens, lg_tokens

# CrossViT (could actually also be used in ViT)
# helper function that makes the embedding from patches
# have a look at the image embedding in ViT
class ImageEmbedder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        dropout = 0.
    ):
        """ImageEmbedder: nn.Module

        Parameters
        ----------
        dim : int
            embedding dimension of the transformer (64, 256 or 512)
        image_size : int
            size of height=width of the input image
        patch_size : int
            size of the image patch obtained by 3*height*width
        dropout : float
            dropout rate applied to the output of the patch embedding
        """
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2 # total number of patches in the image + 1 cls token concatenated later
        patch_dim = 3 * patch_size ** 2

        # create layer that re-arranges the image patches
        # and embeds them with layer norm + linear projection + layer norm
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim), # NOTE: here we project the patch_dim=3*patch_size**2 to dim=64, 256 or 512
            nn.LayerNorm(dim),
        )
        # create/initialize #dim-dimensional positional embedding (will be learned)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) # (num_imgs, num_patches + 1, dim)
        # create #dim cls tokens (for each patch embedding)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # (num_imgs, num_patches, dim) -> (num_imgs, 1, dim) because there is only 1 cls token per image.
        # create dropput layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        # forward through patch embedding layer
        x = self.to_patch_embedding(img) # x: (num_imgs, num_patches, dim)
        # concat class tokens
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0]) # cls_token: (num_imgs, 1, dim)
        x = torch.cat((cls_token, x), dim=1) # x: (num_imgs, num_patches + 1, dim)
        # and add positional embedding
        x += self.pos_embedding
        return self.dropout(x)


# normal ViT
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        """ViT: Vision Transformer

        Parameters
        ----------
        image_size : int or tuple(width, height)
            input image size in pixels
        patch_size : int or tuple(width, height)
            patch size in pixels (usually 16x16 or 8x8) must be divisible by image_size
        num_classes : int
            number of network output classes
        dim : int
            size of image embedding processed by the transformer (usually 256 or 512) but here we use 64
        depth : int
            defines the number of transformer blocks stacked on top of each other
        heads : int
            number of heads in the multi-head self-attention
        mlp_dim : int
            number of units in the feed forward network in transformer blocks
        pool : str, optional
            if 'cls' the class token is used as the image embedding for mlp classification head, however,
            if 'mean' the mean of all patch embeddings is used, by default 'cls'
        channels : int, optional
            number of input channels, by default 3 (RGB)
        dim_head : int, optional
            size of the head in multi-head self-attention (dim_head = dim / heads)
        dropout : float
            dropout rate applied to the output of the transformer blocks
        emb_dropout : float
            dropout rate applied to the output of the (patch embedding and) positional embedding
        """
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # initialize patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        ) # (num_imgs, num_patches, dim) where dim = 3 * patch_size ** 2, and num_patches = (image_size // patch_size) ** 2
        # discards the patches at the border of the image so image size must be divisible by patch size

        # only one positional embedding and one cls token is learnt for all images during the training
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) # (num_imgs, num_patches + 1, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # (num_imgs, num_patches, dim) -> (num_imgs, 1, dim) because there is only 1 cls token per image.
        self.dropout = nn.Dropout(emb_dropout)

        # create transformer blocks
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # create mlp head (layer norm + linear layer)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # patch embedding
        x = self.to_patch_embedding(img) # x: (num_imgs, num_patches, dim)
        b, n, _ = x.shape

        # concat class token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b) # cls_tokens: (num_imgs, 1, dim), however, memory addresses are the same for all images in the batch
        x = torch.cat((cls_tokens, x), dim=1) # x: (num_imgs, num_patches + 1, dim)
        # add positional embedding
        x += self.pos_embedding[:, :(n + 1)] # we learn single positional embedding for all images in the batch. Confirmed by Claude+
        # there's no need to index the positional embedding because it's already the right size. we could have just wrote
        # x += self.pos_embedding
        # apply dropout
        x = self.dropout(x)

        # forward through the transformer blocks
        x = self.transformer(x)

        # decide if x is the mean of the embedding 
        # or the class token
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # transfer via an identity layer the cls tokens or the mean
        # to a latent space, which can then be used as input
        # to the mlp head
        x = self.to_latent(x)
        return self.mlp_head(x)


# CrossViT
class CrossViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        num_classes,
        sm_dim,
        lg_dim,
        sm_patch_size = 12,
        sm_enc_depth = 1,
        sm_enc_heads = 8,
        sm_enc_mlp_dim = 2048,
        sm_enc_dim_head = 64,
        lg_patch_size = 16,
        lg_enc_depth = 4,
        lg_enc_heads = 8,
        lg_enc_mlp_dim = 2048,
        lg_enc_dim_head = 64,
        cross_attn_depth = 2,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        depth = 3,
        dropout = 0.1,
        emb_dropout = 0.1
    ):
        super().__init__()
        # create ImageEmbedder for small and large patches
        self.sm_img_embedder = ImageEmbedder(dim=sm_dim, image_size=image_size, patch_size=sm_patch_size, dropout=emb_dropout)
        self.lg_img_embedder = ImageEmbedder(dim=lg_dim, image_size=image_size, patch_size=lg_patch_size, dropout=emb_dropout)

        # create MultiScaleEncoder
        self.multi_scale_encoder = MultiScaleEncoder(
            depth = depth,
            sm_dim = sm_dim,
            lg_dim = lg_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            sm_enc_params = dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                mlp_dim = sm_enc_mlp_dim,
                dim_head = sm_enc_dim_head
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                mlp_dim = lg_enc_mlp_dim,
                dim_head = lg_enc_dim_head
            ),
            dropout = dropout
        )

        # create mlp heads for small and large patches
        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))

    def forward(self, img):
        # apply image embedders
        sm_tokens = self.sm_img_embedder(img)
        lg_tokens = self.lg_img_embedder(img)

        # and the multi-scale encoder
        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)

        # call the mlp heads w. the class tokens 
        sm_logits = self.sm_mlp_head(sm_tokens[:, 0]) # sm_tokens[:, 0]: (num_imgs, dim) where dim=0 is the cls token
        lg_logits = self.lg_mlp_head(lg_tokens[:, 0]) # lg_tokens[:, 0]: (num_imgs, dim) where dim=0 is the cls token
        
        return sm_logits + lg_logits


if __name__ == "__main__":
    x = torch.randn(16, 3, 32, 32)
    vit = ViT(image_size = 32, patch_size = 8, num_classes = 10, dim = 64, depth = 2, heads = 8, mlp_dim = 128, dropout = 0.1, emb_dropout = 0.1)
    cvit = CrossViT(image_size = 32, num_classes = 10, sm_dim = 64, lg_dim = 128, sm_patch_size = 8,
                    sm_enc_depth = 2, sm_enc_heads = 8, sm_enc_mlp_dim = 128, sm_enc_dim_head = 64,
                    lg_patch_size = 16, lg_enc_depth = 2, lg_enc_heads = 8, lg_enc_mlp_dim = 128,
                    lg_enc_dim_head = 64, cross_attn_depth = 2, cross_attn_heads = 8, cross_attn_dim_head = 64,
                    depth = 3, dropout = 0.1, emb_dropout = 0.1)
    print(vit(x).shape)
    print(cvit(x).shape)
