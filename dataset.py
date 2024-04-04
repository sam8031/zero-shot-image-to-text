class Dataset:
  def __init__(self):
    self.data = self.load_data(5)


  def load_data(self, size):
    data = []
    with open("dataset/captions.txt", 'r') as file:
      next(file)
      i = 0
      for line in file:
        if i == size:
          break
        image_path, caption = line.split(",")
        data.append(("dataset/images/" + image_path, caption))
        i += 1

    return data




