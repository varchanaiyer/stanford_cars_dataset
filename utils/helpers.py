from sklearn.model_selection import train_test_split

def split(data, percentage=0.20):
    x, y = train_test_split(list(data), test_size=percentage, random_state=42)
    X_path, X_class=zip(*x)
    Y_path, Y_class=zip(*y)

    return X_path, X_class, Y_path, Y_class
