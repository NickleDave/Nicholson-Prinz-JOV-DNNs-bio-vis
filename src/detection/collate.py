def frcnn_collate_fn(batch):
    images = [item['img'] for item in batch]
    targets = [item['target'] for item in batch]
    return (images, targets)
