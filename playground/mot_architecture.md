# MoT (Mixture of Transformers) Architecture

## 1. 总体架构对比

### Traditional Transformer vs MoT

```
Traditional Transformer:                    MoT (Mixture of Transformers):
┌─────────────────────┐                     ┌─────────────────────┐
│   Input Sequence    │                     │   Input Sequence    │
│ [T1, T2, I1, I2]    │                     │ [T1, T2, I1, I2]    │
└─────────┬───────────┘                     └─────────┬───────────┘
          │                                           │
          ▼                                           ▼
┌─────────────────────┐                     ┌─────────────────────┐
│   Shared Attention  │                     │  Modality Routing   │
│  (same params for   │                     │  T1,T2→Text Expert  │
│   all tokens)       │                     │  I1,I2→Image Expert │
└─────────┬───────────┘                     └─────────┬───────────┘
          │                                           │
          ▼                                           ▼
┌─────────────────────┐                     ┌─────────────────────┐
│   Shared FFN        │                     │ ModalityUntiedAttn  │
│  (same params for   │                     │ + ModalitySpecific  │
│   all tokens)       │                     │        FFN          │
└─────────────────────┘                     └─────────────────────┘
```

## 2. MoT详细架构图

```
Input: x = [Text_tokens, Image_tokens]  Shape: [batch, seq_len, dim]
       │
       ├─ Modality Masks Creation
       │  ├─ text_mask:  [1,1,0,0,0,0,0,0]
       │  └─ image_mask: [0,0,1,1,1,1,1,1]
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    MoT Transformer Block                        │
│                                                                  │
│  ┌────────────────── ModalityUntiedAttention ─────────────────┐  │
│  │                                                            │  │
│  │  Step 1: Modality-Specific QKV Projection                 │  │
│  │  ┌──────────────┐              ┌──────────────┐           │  │
│  │  │ Text Tokens  │              │ Image Tokens │           │  │
│  │  │   [T1, T2]   │              │[I1,I2,I3,I4] │           │  │
│  │  └──────┬───────┘              └──────┬───────┘           │  │
│  │         │                             │                   │  │
│  │         ▼                             ▼                   │  │
│  │  ┌──────────────┐              ┌──────────────┐           │  │
│  │  │ Text Expert  │              │ Image Expert │           │  │
│  │  │   Wq_text    │              │   Wq_image   │           │  │
│  │  │   Wk_text    │              │   Wk_image   │           │  │
│  │  │   Wv_text    │              │   Wv_image   │           │  │
│  │  └──────┬───────┘              └──────┬───────┘           │  │
│  │         │                             │                   │  │
│  │         ▼                             ▼                   │  │
│  │  ┌──────────────┐              ┌──────────────┐           │  │
│  │  │   Q_text     │              │   Q_image    │           │  │
│  │  │   K_text     │              │   K_image    │           │  │
│  │  │   V_text     │              │   V_image    │           │  │
│  │  └──────┬───────┘              └──────┬───────┘           │  │
│  │         │                             │                   │  │
│  │         └─────────┬───────────────────┘                   │  │
│  │                   │                                       │  │
│  │  Step 2: Merge & Global Attention 🌟                     │  │
│  │                   ▼                                       │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │          Global Q, K, V Tensors                    │ │  │
│  │  │  Q = [Q_text_T1, Q_text_T2, Q_img_I1, Q_img_I2,  │ │  │
│  │  │       Q_img_I3, Q_img_I4]                         │ │  │
│  │  │  K = [K_text_T1, K_text_T2, K_img_I1, K_img_I2,  │ │  │
│  │  │       K_img_I3, K_img_I4]                         │ │  │
│  │  │  V = [V_text_T1, V_text_T2, V_img_I1, V_img_I2,  │ │  │
│  │  │       V_img_I3, V_img_I4]                         │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                   │                                       │  │
│  │                   ▼                                       │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │             Attention Matrix                        │ │  │
│  │  │    Scores = Q @ K.T / √(head_dim)                  │ │  │
│  │  │                                                     │ │  │
│  │  │       T1   T2   I1   I2   I3   I4                  │ │  │
│  │  │  T1 │ ●    ●    ●    ●    ●    ● │ ←Cross-modal   │ │  │
│  │  │  T2 │ ●    ●    ●    ●    ●    ● │   attention!   │ │  │
│  │  │  I1 │ ●    ●    ●    ●    ●    ● │                │ │  │
│  │  │  I2 │ ●    ●    ●    ●    ●    ● │                │ │  │
│  │  │  I3 │ ●    ●    ●    ●    ●    ● │                │ │  │
│  │  │  I4 │ ●    ●    ●    ●    ●    ● │                │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                   │                                       │  │
│  │                   ▼                                       │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │            Attention Output                         │ │  │
│  │  │   AttnOut = Softmax(Scores) @ V                    │ │  │
│  │  └─────────────────┬───────────────────────────────────┘ │  │
│  │                   │                                       │  │
│  │  Step 3: Modality-Specific Output Projection             │  │
│  │                   ▼                                       │  │
│  │         ┌─────────────┐              ┌─────────────┐      │  │
│  │         │ Text Tokens │              │Image Tokens │      │  │
│  │         │ [AttnOut_T] │              │[AttnOut_I]  │      │  │
│  │         └──────┬──────┘              └──────┬──────┘      │  │
│  │                │                            │             │  │
│  │                ▼                            ▼             │  │
│  │         ┌─────────────┐              ┌─────────────┐      │  │
│  │         │ Text Expert │              │Image Expert │      │  │
│  │         │   Wo_text   │              │  Wo_image   │      │  │
│  │         │ Norm_text   │              │ Norm_image  │      │  │
│  │         └──────┬──────┘              └──────┬──────┘      │  │
│  │                │                            │             │  │
│  │                └────────────┬───────────────┘             │  │
│  │                             │                             │  │
│  └─────────────────────────────┼─────────────────────────────┘  │
│                                │                                │
│  ┌─────────────── ModalitySpecificFFN ───────────────────────┐  │
│  │                             │                             │  │
│  │                             ▼                             │  │
│  │         ┌─────────────┐              ┌─────────────┐      │  │
│  │         │ Text Tokens │              │Image Tokens │      │  │
│  │         └──────┬──────┘              └──────┬──────┘      │  │
│  │                │                            │             │  │
│  │                ▼                            ▼             │  │
│  │         ┌─────────────┐              ┌─────────────┐      │  │
│  │         │ Text Expert │              │Image Expert │      │  │
│  │         │  FFN_text   │              │  FFN_image  │      │  │
│  │         │ Norm_text   │              │ Norm_image  │      │  │
│  │         └──────┬──────┘              └──────┬──────┘      │  │
│  │                │                            │             │  │
│  │                └────────────┬───────────────┘             │  │
│  │                             │                             │  │
│  └─────────────────────────────┼─────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │    Final Output         │
                   │ [T1', T2', I1', I2',    │
                   │  I3', I4']              │
                   └─────────────────────────┘
```

## 3. Global Attention的关键作用 🌟

### 3.1 Cross-Modal Information Exchange

```
Before Global Attention:
Text tokens: [T1, T2] (processed by text-specific parameters)
Image tokens: [I1, I2, I3, I4] (processed by image-specific parameters)
↓
Problem: No interaction between modalities!

After Global Attention:
┌─────────────────────────────────────────────────────┐
│         Attention allows every token to            │
│         attend to every other token:               │
│                                                     │
│  T1 ←→ T2: Text-to-text interaction               │
│  T1 ←→ I1,I2,I3,I4: Text-to-image interaction     │
│  I1 ←→ I2,I3,I4: Image-to-image interaction       │
│  I1 ←→ T1,T2: Image-to-text interaction           │
│                                                     │
│  🎯 This enables cross-modal understanding!        │
└─────────────────────────────────────────────────────┘
```

### 3.2 Attention Matrix详解

```
        Query Tokens
        T1  T2  I1  I2  I3  I4
    T1 │0.8 0.6 0.3 0.2 0.1 0.2│ ← T1 mostly attends to text
K   T2 │0.7 0.9 0.1 0.1 0.0 0.1│ ← T2 focuses on T1,T2
e   I1 │0.3 0.2 0.8 0.7 0.6 0.5│ ← I1 attends to images + some text
y   I2 │0.2 0.1 0.6 0.9 0.8 0.7│ ← I2 focuses on images
    I3 │0.1 0.0 0.5 0.6 0.9 0.8│ ← I3 mostly attends to images
    I4 │0.2 0.1 0.4 0.5 0.7 0.9│ ← I4 focuses on nearby images

Cross-modal connections: T1↔I1 (0.3), T2↔I1 (0.2), etc.
```

## 4. MoT的核心优势

### 4.1 模态专用化 + 跨模态交互

```
┌────────────────────┐    Global    ┌────────────────────┐
│   Text Expert      │  Attention   │   Image Expert     │
│  ┌──────────────┐  │◄──────────►  │  ┌──────────────┐  │
│  │ Specialized  │  │              │  │ Specialized  │  │
│  │ for text     │  │              │  │ for images   │  │
│  │ patterns     │  │              │  │ patterns     │  │
│  └──────────────┘  │              │  └──────────────┘  │
└────────────────────┘              └────────────────────┘

Best of both worlds:
✅ Specialized processing per modality
✅ Cross-modal information exchange
```

### 4.2 参数效率对比

```
Traditional Transformer:
- Shared parameters for all modalities
- May not capture modality-specific patterns well

MoT:
- 2x parameters (one set per modality)
- But much better performance on multimodal tasks
- Each modality gets optimized representations
```

## 5. 代码中的体现

### 5.1 模态路由

```python
# Step 1: Route tokens to modality experts
for i in range(self.n_modalities):
    mask = modality_masks[i]  # [True, True, False, False] for text
    expert_input = x[mask]    # Extract text tokens
    
    # Apply modality-specific projections
    xq = self.local_experts_wq[i](expert_input)  # Text-specific Wq
    xk = self.local_experts_wk[i](expert_input)  # Text-specific Wk  
    xv = self.local_experts_wv[i](expert_input)  # Text-specific Wv
```

### 5.2 Global Attention

```python
# Step 2: Merge and compute global attention
xq = merge_modalities(expert_outputs_xq, modality_masks, target_shape)
xk = merge_modalities(expert_outputs_xk, modality_masks, target_shape)
xv = merge_modalities(expert_outputs_xv, modality_masks, target_shape)

# Global attention computation - ALL tokens interact!
scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(head_dim)
attn_weights = F.softmax(scores, dim=-1)
attn_output = torch.matmul(attn_weights, xv)
```

### 5.3 模态专用输出处理

```python
# Step 3: Route outputs back to modality experts
for i in range(self.n_modalities):
    mask = modality_masks[i]
    expert_input = attn_output[mask]
    
    # Apply modality-specific output projection and normalization
    expert_output = self.local_experts_wo[i](expert_input)
    expert_output = self.local_experts_attention_norm[i](expert_output)
    final_output[mask] = expert_output
```
