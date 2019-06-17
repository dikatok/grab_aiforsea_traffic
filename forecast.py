import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to model directory', default="models")
    parser.add_argument('--csv', help='Path to csv file to be forecast', required=True)
    args = parser.parse_args()

    models_path = args.model
    csv_path = args.csv

    print(models_path, csv_path)
