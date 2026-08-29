from ingest.source import flatten


def test_the_entry_point_is_the_file_opening_the_document():
    members = {
        "macros": r"\newcommand{\x}{y}",
        "main": r"\begin{document}Body\end{document}",
    }

    assert "Body" in flatten(members)


def test_inputs_are_resolved_in_place():
    """A submission can hold a dozen fragments and only one preamble."""
    members = {
        "main": r"\begin{document}\input{results}\end{document}",
        "results": "Measured value is 0.42.",
    }

    assert "Measured value is 0.42." in flatten(members)


def test_include_is_resolved_like_input():
    members = {
        "main": r"\begin{document}\include{part}\end{document}",
        "part": "Included text.",
    }

    assert "Included text." in flatten(members)


def test_a_missing_target_is_skipped_rather_than_raising():
    members = {"main": r"\begin{document}\input{absent}Body\end{document}"}

    assert "Body" in flatten(members)


def test_a_cycle_terminates():
    """Two fragments that include each other would otherwise recurse forever."""
    members = {
        "main": r"\begin{document}\input{a}\end{document}",
        "a": r"\input{b}A",
        "b": r"\input{a}B",
    }

    assert flatten(members).count("A") == 1


def test_a_submission_without_a_document_yields_nothing():
    assert flatten({"notes": "just some macros"}) == ""


def test_the_extension_is_optional_in_the_reference():
    members = {
        "main": r"\begin{document}\input{section.tex}\end{document}",
        "section": "Section body.",
    }

    assert "Section body." in flatten(members)
