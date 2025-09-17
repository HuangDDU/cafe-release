def method_info(name, version="0.0.1", description="", **kwargs):
    def decorator(func):
        func._method_info = {"name": name, "version": version, "description": description}
        func._method_info.update(kwargs)
        return func

    return decorator
