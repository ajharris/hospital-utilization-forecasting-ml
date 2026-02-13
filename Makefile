.PHONY: install ingest train backtest

install:
\tpip install -r requirements.txt

ingest:
\tpython -m src.ingestion
\tpython -m src.validation
\tpython -m src.features

train:
\tpython -m src.train

backtest:
\tpython -m src.backtest
