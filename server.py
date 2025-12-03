import flwr as fl

def main():

    strategy = fl.server.strategy.FedAvg(
        on_fit_config_fn=lambda rnd: {"server_round": rnd},
        on_evaluate_config_fn=lambda rnd: {"server_round": rnd},
    )

    config = fl.server.ServerConfig(num_rounds=2)

    fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=config,
        strategy=strategy,
    )

if __name__ == "__main__":
    main()

