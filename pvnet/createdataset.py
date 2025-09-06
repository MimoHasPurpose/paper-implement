from datasets.linemod_dataset import LineMODDataset
ds = LineMODDataset("./LINEMOD/cat", input_size=480, training=False)
print("Dataset size:", len(ds))
s = ds[0]
print("image:", s['image'].shape)
print("vec_gt:", s['vec_gt'].shape)
print("mask_s:", s['mask_s'].shape)
print("kpts2d:", s['kpts2d'].shape)