from ucimlrepo import fetch_ucirepo


def get_data():
    return fetch_ucirepo(id=336)


if __name__ == '__main__':
    df = get_data().data.original
    df.to_csv('cdk.csv')
