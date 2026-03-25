def classFactory(iface):
    from .connect import ConnectPlugin
    return ConnectPlugin(iface)
