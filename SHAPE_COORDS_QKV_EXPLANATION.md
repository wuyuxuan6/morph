# Shape Stage: `coords` And `Q/K/V`

这份说明只解释一件事：

在 `MorphAny3D2` 的 `shape` 阶段里，`coords` 到底怎么进入模型，以及它如何影响 `Q/K/V` 和 attention。

相关代码位置：

- pipeline 入口：
  [trellis2_image_to_3d.py](/data/wyx/Projects/MorphAny3D2/trellis2/pipelines/trellis2_image_to_3d.py)
- `shape` flow model：
  [structured_latent_flow.py](/data/wyx/Projects/MorphAny3D2/trellis2/models/structured_latent_flow.py)
- sparse transformer block：
  [modulated.py](/data/wyx/Projects/MorphAny3D2/trellis2/modules/sparse/transformer/modulated.py)
- sparse attention：
  [modules.py](/data/wyx/Projects/MorphAny3D2/trellis2/modules/sparse/attention/modules.py)
- sparse rope：
  [rope.py](/data/wyx/Projects/MorphAny3D2/trellis2/modules/sparse/attention/rope.py)

## 1. `shape` 阶段的输入不是“只有噪声”

`shape` 阶段的输入是一个 `SparseTensor`，包含两部分：

- `coords`
- `feats`

在 pipeline 里是这样构造的：

```python
noise = SparseTensor(
    feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
    coords=coords,
)
```

位置：
[trellis2_image_to_3d.py](/data/wyx/Projects/MorphAny3D2/trellis2/pipelines/trellis2_image_to_3d.py#L346)

这里含义非常明确：

- `coords` 决定 token 在 3D 空间的哪些位置存在
- `feats` 是这些位置上的随机初始 feature

所以 `shape` 输入不是“没有位置的纯噪声”，而是：

`固定空间坐标上的随机 sparse feature`

如果开启 morphing 初始化插值，则 `feats` 不再是随机值，而是 source/target cache 对齐后插值得到的 feature，但 `coords` 仍然来自当前阶段的结构结果。

## 2. `coords` 从哪来

`shape` 之前先跑 `sparse structure`，得到一批结构坐标：

```python
coords, _, _ = self.sample_sparse_structure_morphing(...)
shape_slat = self.sample_shape_slat_morphing(..., coords, ...)
```

位置：
[trellis2_image_to_3d.py](/data/wyx/Projects/MorphAny3D2/trellis2/pipelines/trellis2_image_to_3d.py#L833)

所以 `shape` 阶段不负责重新决定 token 的空间分布，它直接继承第一阶段给出的稀疏 3D 坐标。

## 3. `coords` 先作为 SparseTensor 的空间索引保留下来

进入 `shape` flow model 后，输入类型是：

```python
def forward(self, x: sp.SparseTensor, ...)
```

位置：
[structured_latent_flow.py](/data/wyx/Projects/MorphAny3D2/trellis2/models/structured_latent_flow.py#L169)

第一步是：

```python
h = self.input_layer(x)
```

位置：
[structured_latent_flow.py](/data/wyx/Projects/MorphAny3D2/trellis2/models/structured_latent_flow.py#L186)

这一步只对 `feats` 做通道投影，不会丢掉 `coords`。

也就是说，经过输入层后：

- `h.feats` 变成 transformer 隐藏维度
- `h.coords` 仍然保留原来的 3D 坐标

所以从这里开始，每个 token 一直同时带着：

- 内容特征
- 空间位置

## 4. `coords` 会一路带进每个 transformer block

后面每个 block 直接吃 `SparseTensor h`：

```python
for block_idx, block in enumerate(self.blocks):
    h = block(h, t_emb, cond, step_idx, block_idx, **kwargs)
```

位置：
[structured_latent_flow.py](/data/wyx/Projects/MorphAny3D2/trellis2/models/structured_latent_flow.py#L201)

这意味着 block 里拿到的不是普通 dense tensor，而是：

- `x.feats`
- `x.coords`

一起。

## 5. `Q/K/V` 的数值来自 `feats`，但它们仍然带着 `coords`

在 sparse self-attention 里：

```python
qkv = self._linear(self.to_qkv, x)
qkv = self._fused_pre(qkv, num_fused=3)
```

位置：
[modules.py](/data/wyx/Projects/MorphAny3D2/trellis2/modules/sparse/attention/modules.py#L111)

这两行的作用是：

- 用 `x.feats` 线性投影出 `qkv`
- 但 `qkv` 仍然是带 `coords` 的 sparse 表示

后面会拆开：

```python
q, k, v = qkv.unbind(dim=-3)
```

位置：
[modules.py](/data/wyx/Projects/MorphAny3D2/trellis2/modules/sparse/attention/modules.py#L157)

所以：

- `Q/K/V` 的数值来自当前 hidden feature 的线性投影
- `Q/K/V` 的 token 空间位置仍然由 `coords` 标识

换句话说：

- `feats` 决定“这个 token 当前表示什么”
- `coords` 决定“这个 token 在空间中是谁”

## 6. 在 self-attention 的 RoPE 里，`coords` 直接改变 `Q/K`

这是 `coords` 最关键的作用点。

在 sparse self-attention 里，如果启用 rope：

```python
if self.use_rope:
    q, k = self.rope(q, k)
```

位置：
[modules.py](/data/wyx/Projects/MorphAny3D2/trellis2/modules/sparse/attention/modules.py#L161)

进入 `rope.py` 后，真正使用的是 token 的三维坐标：

```python
coords = tensor.coords[..., 1:]
phases = self._get_phases(coords.reshape(-1)).reshape(*coords.shape[:-1], -1)
```

位置：
[rope.py](/data/wyx/Projects/MorphAny3D2/trellis2/modules/sparse/attention/rope.py#L41)

然后对 `Q/K` 做旋转：

```python
x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
x_rotated = x_complex * phases.unsqueeze(-2)
```

位置：
[rope.py](/data/wyx/Projects/MorphAny3D2/trellis2/modules/sparse/attention/rope.py#L24)

这意味着：

- `coords` 不只是生成一个附加 embedding
- `coords` 会直接决定 `Q/K` 在特征空间里如何旋转

所以在 `shape` 的 self-attention 中：

`coords` 直接进入 attention 的匹配规则

这也是为什么 `shape TFSA` 在 `rope` 下更敏感。

## 7. cross-attention 里，`coords` 作用于当前 `Q` 这一边

cross-attention 的代码是：

```python
q = self._linear(self.to_q, x)
kv = self._linear(self.to_kv, context)
```

位置：
[modules.py](/data/wyx/Projects/MorphAny3D2/trellis2/modules/sparse/attention/modules.py#L176)

这里：

- `q` 来自当前 `shape` token
- `kv` 来自图像条件 `context`

而模块初始化时就限定了 cross-attention 不用 rope：

```python
assert type == "self" or use_rope is False
```

位置：
[modules.py](/data/wyx/Projects/MorphAny3D2/trellis2/modules/sparse/attention/modules.py#L59)

所以在 cross-attention 里：

- `coords` 不会去给图像条件 token 编 sparse 3D rope
- 但 `coords` 仍然决定当前 sparse token 的身份，也就是决定当前 `Q` 来自哪一批 3D token

## 8. TFSA 里，`coords` 还决定如何从上一帧 cache 对齐 `K/V`

这部分是理解 `shape TFSA` 的核心。

当前实现里先按坐标做最近邻对齐：

```python
def align_cached_kv_to_query(q, cache_coords, cache_k, cache_v):
    query_xyz = q.coords[:, 1:].float()
    cache_xyz = cache_coords[:, 1:].float()
    indices = torch.argmin(torch.cdist(query_xyz, cache_xyz, p=2), dim=1)
    aligned_k = cache_k[indices]
    aligned_v = cache_v[indices]
    return q.coords, aligned_k, aligned_v
```

位置：
[modules.py](/data/wyx/Projects/MorphAny3D2/trellis2/modules/sparse/attention/modules.py#L10)

然后在 TFSA 分支里：

```python
aligned_coords, aligned_k, aligned_v = align_cached_kv_to_query(...)
cached_k = SparseTensor(feats=aligned_k, coords=aligned_coords)
cached_v = SparseTensor(feats=aligned_v, coords=aligned_coords)
```

位置：
[modules.py](/data/wyx/Projects/MorphAny3D2/trellis2/modules/sparse/attention/modules.py#L139)

接着再做 rope：

```python
if self.use_rope:
    q, cached_k = self.rope(q, cached_k)
```

位置：
[modules.py](/data/wyx/Projects/MorphAny3D2/trellis2/modules/sparse/attention/modules.py#L150)

所以在 TFSA 里，`coords` 有两层作用：

1. 用当前帧坐标去上一帧 cache 坐标里找对应 token
2. 再让当前 `q` 和对齐后的 `cached_k` 各自按自己的坐标做 rope

这就是为什么 `coords` 不只是“位置标签”，而是 TFSA 是否成立的前提。

## 9. 只有在 APE 模式下，`coords` 才是“先转成位置 embedding 再加到 hidden state”

在同一个 flow model 中，如果 `pe_mode == "ape"`：

```python
pe = self.pos_embedder(h.coords[:, 1:])
h = h + pe
```

位置：
[structured_latent_flow.py](/data/wyx/Projects/MorphAny3D2/trellis2/models/structured_latent_flow.py#L199)

这时 `coords` 的作用是：

- 先生成绝对位置 embedding
- 再加到 hidden state 上

但 `TRELLIS.2` 的 `shape` 分支主要是 `rope`，所以当前 `shape TFSA` 的问题主要不是这条路径。

## 10. 最后把整条链压缩成一句话

在 `MorphAny3D2` 的 `shape` 阶段里，`coords` 的作用有三层：

1. 决定有哪些 sparse token，以及它们在 3D 空间中的位置
2. 在 self-attention 的 RoPE 里直接决定 `Q/K` 如何旋转
3. 在 TFSA 中决定上一帧哪些 `K/V` 会被对齐并复用到当前帧

所以 `coords` 不是附属信息，而是：

`shape token 的空间身份 + self-attention 的几何编码依据 + 跨帧 cache 对齐依据`

