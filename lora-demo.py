def mark_only_lora_as_trainable(model):
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 解冻LoRA参数（通常命名为lora_A或lora_B）
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
            

clip_config:
  params:
    lora_adapt: True  # 启用LoRA
    rank: 16          # 低秩矩阵的秩，值越小参数量越少

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=learning_rate
)