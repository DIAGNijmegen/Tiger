import re

import tiger.io


def test_read_write_json(tmp_path):
    """Test writing to and reading from JSON files"""
    json_file = tmp_path / "test.json"
    assert not json_file.exists()

    written_content = {"foo": [1, 2, 3]}
    written_json = '{"foo":[1,2,3]}'

    # Test writing
    tiger.io.write_json(json_file, written_content)
    with open(str(json_file)) as fp:
        read_json = re.sub(r"\s", "", fp.read())  # ignore whitespace and line breaks in the file
    assert read_json == written_json

    # Test reading
    read_content = tiger.io.read_json(json_file)
    assert read_content == written_content

    # Cleanup
    json_file.unlink()
