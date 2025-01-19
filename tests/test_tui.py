from lasagna.tui import _truncate_str


def test_truncate_str():
    # String is shorter than allowance:
    t = _truncate_str('hi', 10)
    assert t == 'hi'

    # String equals the allowance:
    t = _truncate_str('hi', 2)
    assert t == 'hi'

    # String is less than allowance, but showing the
    # truncated message would be *longer*:
    t = _truncate_str('hi', 1)
    assert t == 'hi'

    lorem_ipsum = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi sed lacus nisl. Aliquam pharetra lacus non diam porttitor, quis congue libero imperdiet. Vestibulum dapibus est ut urna interdum, eu sagittis metus lobortis. Ut quis tempor dolor, vitae sagittis lectus. Morbi non ex vel leo ullamcorper hendrerit. In eleifend, tortor tincidunt mattis mollis, nibh mauris ultricies velit, convallis fringilla lectus dui non dui. Duis quis elit vitae elit pretium efficitur. Fusce urna magna, interdum quis interdum sit amet, congue a magna. Maecenas vitae aliquam purus. Aliquam erat volutpat. Etiam sit amet pretium massa.'

    # String is shorter than allowance:
    t = _truncate_str(lorem_ipsum, 10000000)
    assert t == lorem_ipsum

    # String equals the allowance:
    t = _truncate_str(lorem_ipsum, len(lorem_ipsum))
    assert t == lorem_ipsum

    # String is less than allowance, but showing the
    # truncated message would be *longer*:
    t = _truncate_str(lorem_ipsum, len(lorem_ipsum) - 5)
    assert t == lorem_ipsum

    # Truncation is necessary:
    t = _truncate_str(lorem_ipsum, 11)
    assert t == 'Lorem ipsum [... truncated 607 characters ...]'
