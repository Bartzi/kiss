

class ChainUpdateDisabler:

    def __init__(self, chains):
        self.chains = chains

    def __enter__(self):
        for chain in self.chains:
            chain.disable_update()
            chain.cleargrads()

    def __exit__(self, type, value, tb):
        for chain in self.chains:
            chain.enable_update()
            chain.cleargrads()


def disable_chains(chains):
    return ChainUpdateDisabler(chains)
