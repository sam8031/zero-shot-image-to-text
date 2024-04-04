class Dataset:
  def __init__(self, size, path):
    self.data = self.load_data(size, path)

  def __len__(self):
    return len(self.data[0])

  def load_data(self, size, path):
    image_paths = []
    captions = []
    data = (image_paths, captions)
    with open(path, 'r') as file:
      next(file)
      i = 0
      for line in file:
        if i == size:
          break
        image_path, caption = line.split(",", 1)
        if image_path not in image_paths:
          image_paths.append("dataset/images/" + image_path)
        captions.append(caption)
        i += 1

    return data




