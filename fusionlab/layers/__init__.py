from fusionlab import BACKEND
if BACKEND['torch']:
    from .factories import (
        ConvND,
        ConvT,
        Upsample,
        BatchNorm,
        InstanceNorm,
        MaxPool,
        AvgPool,
        AdaptiveMaxPool,
        AdaptiveAvgPool,
        ReplicationPad,
        ConstantPad
    )
    from .squeeze_excitation.se import SEModule
    from .base import (
        ConvNormAct,
        Rearrange,
        DropPath,
    )
    from .patch_embed.patch_embedding import PatchEmbedding
    from .selfattention.selfattention import (
        SelfAttention,
        SRAttention,
    )
if BACKEND['tf']:
    from .squeeze_excitation.tfse import TFSEModule