class Dataset:
  def __init__(self):
    self.data = self.load_data(5)

  def __len__(self):
    return len(self.data[0])

  def load_data(self, size):
    image_paths = []
    captions = []
    data = (image_paths, captions)
    with open("dataset/captions.txt", 'r') as file:
      next(file)
      i = 0
      for line in file:
        if i == size:
          break
        image_path, caption = line.split(",")
        image_paths.append("dataset/images/" + image_path)
        captions.append(caption)
        i += 1

    return data




