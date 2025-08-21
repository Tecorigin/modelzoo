def load_data_list(self) -> List[dict]:
    """Load data list."""
    img_prefix = self.data_prefix['img_path']
    annotations_data = mmengine.load(self.ann_file)
    file_backend = get_file_backend(img_prefix)

    # 从标准COCO格式中提取annotations和images
    annotations = annotations_data.get('annotations', [])
    images = annotations_data.get('images', [])
    
    # 构建image_id到文件名的映射
    image_id_to_file = {img['id']: img['file_name'] for img in images}
    
    data_list = []
    
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
            
        # 获取image_id
        image_id = ann.get('image_id')
        if image_id is None:
            continue
            
        # 获取对应的图像文件名
        img_file = image_id_to_file.get(image_id)
        if img_file is None:
            # 如果找不到对应的图像文件，使用默认命名方式
            img_file = f"{image_id:012d}.jpg"
            
        # 构建完整的图像路径
        img_path = file_backend.join_path(img_prefix, img_file)
        
        # 获取caption
        caption = ann.get('caption', '')
        
        data_list.append({
            'image_id': str(image_id),
            'img_path': img_path,
            'gt_caption': caption
        })
    
    return data_list