import pandas as pd

def generate_data() -> pd.DataFrame:
    rows = [
        (0, 0, 200),
        (0, 0, 120),
        (0, 1, 300),
        (1, 0, 500),
        (1, 0, 600),
        (1, 1, 800),
    ]
    # print the content of pd
    return pd.DataFrame(rows, columns=['t', 'x', 'y'])

def main():
    print("Hello from causal-inference-in-python-code!")


if __name__ == "__main__":
    main()
