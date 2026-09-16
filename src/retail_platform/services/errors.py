class NotFoundError(LookupError):
    pass


class ForbiddenError(PermissionError):
    pass


class ConflictError(RuntimeError):
    pass
