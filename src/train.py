import typer

from classification.train.pipeline import run


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
