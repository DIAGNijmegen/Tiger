from tiger import TigerException


def test_exception():
    # Without message
    try:
        raise TigerException()
    except TigerException as e:
        assert e.message is None

    # With message
    message = "Hello world"
    try:
        raise TigerException(message)
    except TigerException as e:
        assert e.message == message
