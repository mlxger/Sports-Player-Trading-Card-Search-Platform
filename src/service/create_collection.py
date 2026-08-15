from retrieval import MilvusVectorStore
from settings import get_settings


def main() -> None:
    MilvusVectorStore.create_collection(get_settings())


if __name__ == "__main__":
    main()
