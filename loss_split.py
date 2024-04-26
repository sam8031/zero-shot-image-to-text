import re


def extract_losses(input_file, output_file):
    losses = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            match = re.search(r'Average Loss for Epoch \d+: ([0-9]+\.[0-9]{5,})', line)
            if match:
                loss = match.group(1)
                losses.append(loss)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(',\n'.join(losses))


input_file_path = r"./zero-shot-image-to-text/output1.txt"
output_file_path = r"C:\Users\danie\PycharmProjects\script line losses\losses.txt"
extract_losses(input_file_path, output_file_path)
