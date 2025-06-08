import torch
import torch.nn.functional as F

def clip_loss(img_embs, text_embs, logit_scale):
    """
    标准CLIP对比损失
    
    Args:
        img_embs: 图像嵌入 [batch_size, embed_dim]
        text_embs: 文本嵌入 [batch_size, embed_dim]
        logit_scale: 温度参数的指数
    
    Returns:
        loss: 对比损失
        accuracy: 准确率
    """
    # 归一化嵌入
    batch_size = img_embs.shape[0]
    labels = torch.arange(batch_size, device=img_embs.device).long()

    # 计算相似度
    img_text_similarity = logit_scale * img_embs @ text_embs.t()
    text_img_similarity = logit_scale * text_embs @ img_embs.t()

    # 计算准确率
    preds_i2t = img_text_similarity.argmax(dim=-1)
    preds_t2i = text_img_similarity.argmax(dim=-1)
    acc_i2t = (preds_i2t == labels).float().mean().item()
    acc_t2i = (preds_t2i == labels).float().mean().item()
    accuracy = (acc_i2t + acc_t2i) / 2

    # 计算损失
    loss = (
        F.cross_entropy(img_text_similarity, labels)
        + F.cross_entropy(text_img_similarity, labels)
    ).div(2)
    
    return loss, accuracy

def negclip_loss(img_embs, text_embs, neg_text_embs, logit_scale):
    """
    带负样本的CLIP对比损失
    
    Args:
        img_embs: 图像嵌入 [batch_size, embed_dim]
        text_embs: 文本嵌入 [batch_size, embed_dim]
        neg_text_embs: 负样本文本嵌入 [batch_size, embed_dim]
        logit_scale: 温度参数的指数
    
    Returns:
        loss: 对比损失
        accuracy: 准确率
    """
    # 归一化嵌入
    batch_size = img_embs.shape[0]
    labels = torch.arange(batch_size, device=img_embs.device).long()

    # 计算相似度
    img_text_similarity = logit_scale * img_embs @ text_embs.t()
    text_img_similarity = logit_scale * text_embs @ img_embs.t()
    img_negtext_similarity = logit_scale * img_embs @ neg_text_embs.t()

    # 计算准确率
    preds_i2t = torch.cat((img_text_similarity, img_negtext_similarity), dim=-1).argmax(dim=-1)
    preds_t2i = img_text_similarity.t().argmax(dim=-1)
    acc_i2t = (preds_i2t == labels).float().mean().item()
    acc_t2i = (preds_t2i == labels).float().mean().item()
    accuracy = (acc_i2t + acc_t2i) / 2

    # 计算损失
    loss = (
        F.cross_entropy(
            torch.cat([img_text_similarity, img_negtext_similarity], dim=-1), labels
        )
        + F.cross_entropy(text_img_similarity, labels)
    ).div(2)
    
    return loss, accuracy

def L_caption_neg(
    pos_caption_embs, pos_img_embs,
    neg_caption_embs, neg_img_embs,
    logit_scale
):
    """
    Caption的negclip损失
    
    Args:
        pos_caption_embs: 正样本文本嵌入 [batch_size, embed_dim]
        pos_img_embs: 正样本图像嵌入 [batch_size, embed_dim]
        neg_caption_embs: 负样本文本嵌入 [batch_size, embed_dim]
        neg_img_embs: 负样本图像嵌入 [batch_size, embed_dim]
        logit_scale: 温度参数的指数
    
    Returns:
        loss: 对比损失
        accuracy: 准确率
    """
    # 第一部分：pos_img, pos_caption, neg_caption
    loss_1, accuracy1 = negclip_loss(pos_img_embs, pos_caption_embs, neg_caption_embs, logit_scale)
    
    # 第二部分：neg_img, neg_caption, pos_caption
    loss_2, accuracy2 = negclip_loss(neg_img_embs, neg_caption_embs, pos_caption_embs, logit_scale)
    
    loss = loss_1 + loss_2
    accuracy = (accuracy1 + accuracy2) / 2
    return loss, accuracy

def L_concept_neg(
    pos_concept_embs, pos_img_embs,
    neg_concept_embs, neg_img_embs,
    logit_scale
):
    """
    Concept的negclip损失
    
    Args:
        pos_concept_embs: 正样本概念嵌入 [batch_size, embed_dim]
        pos_img_embs: 正样本图像嵌入 [batch_size, embed_dim]
        neg_concept_embs: 负样本概念嵌入 [batch_size, embed_dim]
        neg_img_embs: 负样本图像嵌入 [batch_size, embed_dim]
        logit_scale: 温度参数的指数
    
    Returns:
        loss: 对比损失
        accuracy: 准确率
    """
    # 第一部分：pos_img, pos_concept, neg_concept
    loss_1, accuracy1 = negclip_loss(pos_img_embs, pos_concept_embs, neg_concept_embs, logit_scale)
    
    # 第二部分：neg_img, neg_concept, pos_concept
    loss_2, accuracy2 = negclip_loss(neg_img_embs, neg_concept_embs, pos_concept_embs, logit_scale)
    
    loss = loss_1 + loss_2
    accuracy = (accuracy1 + accuracy2) / 2
    return loss, accuracy

# ============= 消融实验损失函数 =============

# 1. Caption的CLIP baseline
def ablation_caption_clip(
    pos_caption_embs, pos_img_embs,
    logit_scale
):
    """Caption的CLIP baseline损失"""
    return clip_loss(pos_img_embs, pos_caption_embs, logit_scale)

# 2. Concept的CLIP baseline
def ablation_concept_clip(
    pos_concept_embs, pos_img_embs,
    logit_scale
):
    """Concept的CLIP baseline损失"""
    return clip_loss(pos_img_embs, pos_concept_embs, logit_scale)

# 3. Caption的negclip
def ablation_caption_negclip(
    pos_caption_embs, pos_img_embs,
    neg_caption_embs, neg_img_embs,
    logit_scale
):
    """Caption的negclip损失"""
    return L_caption_neg(
        pos_caption_embs, pos_img_embs,
        neg_caption_embs, neg_img_embs,
        logit_scale
    )

# 4. Concept的negclip
def ablation_concept_negclip(
    pos_concept_embs, pos_img_embs,
    neg_concept_embs, neg_img_embs,
    logit_scale
):
    """Concept的negclip损失"""
    return L_concept_neg(
        pos_concept_embs, pos_img_embs,
        neg_concept_embs, neg_img_embs,
        logit_scale
    )

# 5. Caption的clip + concept的clip
def ablation_caption_concept_clip(
    pos_concept_embs, pos_caption_embs, pos_img_embs,
    logit_scale, lambda_caption=0.5, lambda_concept=0.5
):
    """Caption的clip + concept的clip损失"""
    caption_loss_val, caption_acc = clip_loss(pos_img_embs, pos_caption_embs, logit_scale)
    concept_loss_val, concept_acc = clip_loss(pos_img_embs, pos_concept_embs, logit_scale)
    
    total_loss = lambda_caption * caption_loss_val + lambda_concept * concept_loss_val
    total_accuracy = (caption_acc + concept_acc) / 2
    return total_loss, total_accuracy

# 6. Caption的negclip + concept的clip
def ablation_caption_neg_concept_clip(
    pos_concept_embs, pos_caption_embs, pos_img_embs,
    neg_caption_embs, neg_img_embs,
    logit_scale, lambda_caption=0.5, lambda_concept=0.5
):
    """Caption的negclip + concept的clip损失"""
    caption_loss_val, caption_acc = L_caption_neg(
        pos_caption_embs, pos_img_embs,
        neg_caption_embs, neg_img_embs,
        logit_scale
    )
    
    concept_loss_val, concept_acc = clip_loss(pos_img_embs, pos_concept_embs, logit_scale)
    
    total_loss = lambda_caption * caption_loss_val + lambda_concept * concept_loss_val
    total_accuracy = (caption_acc + concept_acc) / 2
    return total_loss, total_accuracy

# 7. Caption的clip + concept的negclip
def ablation_caption_concept_neg(
    pos_concept_embs, pos_caption_embs, pos_img_embs,
    neg_concept_embs, neg_img_embs,
    logit_scale, lambda_caption=0.5, lambda_concept=0.5
):
    """Caption的clip + concept的negclip损失"""
    caption_loss_val, caption_acc = clip_loss(pos_img_embs, pos_caption_embs, logit_scale)
    
    concept_loss_val, concept_acc = L_concept_neg(
        pos_concept_embs, pos_img_embs,
        neg_concept_embs, neg_img_embs,
        logit_scale
    )
    
    total_loss = lambda_caption * caption_loss_val + lambda_concept * concept_loss_val
    total_accuracy = (caption_acc + concept_acc) / 2
    return total_loss, total_accuracy

# 8. Caption的negclip + concept的negclip (完整的CultureCLIP)
def cultureclip_loss(
    pos_concept_embs, pos_caption_embs, pos_img_embs,
    neg_concept_embs, neg_caption_embs, neg_img_embs,
    logit_scale, lambda_caption=0.5, lambda_concept=0.5
):
    """完整的CultureCLIP损失：Caption的negclip + concept的negclip"""
    caption_loss_val, caption_acc = L_caption_neg(
        pos_caption_embs, pos_img_embs,
        neg_caption_embs, neg_img_embs,
        logit_scale
    )
    
    concept_loss_val, concept_acc = L_concept_neg(
        pos_concept_embs, pos_img_embs,
        neg_concept_embs, neg_img_embs,
        logit_scale
    )
    
    total_loss = lambda_caption * caption_loss_val + lambda_concept * concept_loss_val
    total_accuracy = (caption_acc + concept_acc) / 2
    return total_loss, total_accuracy