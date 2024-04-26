import csv

INPUT_FILE = "./dataset/results_new.csv"
TRAIN_FILE = "./dataset/training.csv"
TEST_FILE = "./dataset/testing.csv"
TRAIN_PERCENTAGE = 0.7
TEST_PERCENTAGE = 0.3

def setup_data():
    # Split the input data into training and testing sets and write to CSV files
    with open(INPUT_FILE, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')
        next(reader)  # Skip the header row
        data = [row for row in reader]

    # Split the data into training and testing sets
    train_size = int(len(data) * TRAIN_PERCENTAGE)
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Write training data to CSV
    with open(TRAIN_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='|')
        writer.writerow(['image_name', 'comment_number', 'comment'])  # Header row
        writer.writerows(train_data)

    # Write testing data to CSV
    with open(TEST_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='|')
        writer.writerow(['image_name', 'comment_number', 'comment'])  # Header row
        writer.writerows(test_data)

    print("Data split into training and testing sets successfully.")

if __name__ == '__main__':
    setup_data()
