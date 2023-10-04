class Register:
    def __init__(self, readonly: bool, label: str) -> None:
        self.readonly = readonly
        self.label = label

    def __get__(self, instance, owner):  # type: ignore
        if instance is None:
            return self
        value = instance._state[self.label]
        return value

    def __set__(self, instance, value):  # type: ignore
        if self.readonly:
            raise AttributeError("can't set attribute")
        instance._state[self.label] = value
