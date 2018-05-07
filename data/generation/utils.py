# TODO: I need to create a regex that searches for latex tokens to ignore
# TODO: I need to create { token => id } and [token,...]

import re

bad_latex_tokens_regex = re.compile(r"""
    \\([a-zA-Z0-9_])*cite([a-zA-Z0-9_])*{.*}
    | \\footnote(\[.*\])?{.*}
    | \\\\*
    | \\kill
    | \\pagebreak
    | \\today
    | \\[hv]space{.*}
    | \\rule{.*}{.*}
    | \\(label|ref|pageref|footnote){.*}
    | \\begin{\s*(table|figure)\s*}
    | \\caption{.*}
""", re.VERBOSE)

inline_math_regex = re.compile(r"""
    \$                      # start inline math expression
    (?P<math_content>.*)    # the content of math mode
    \$                      # end inline math expression
""", re.VERBOSE | re.DOTALL)

math_block_regex = re.compile(r"""
    ^   # start of string
    (\\begin{\s*(equation|multiline|align|eqnarray|IEEEeqnarray)(\s*\*)?\s*}| \$\$)       # starting math mode
    (?P<math_content>.*)  # the content of math mode
    (\\end{\s*(equation|multiline|align|eqnarray|IEEEeqnarray)(\s*\*)?\s*} | \$\$)         # ending math mode
    $   # end of string
""", re.VERBOSE | re.DOTALL)

begin_latex_document = "\\begin{document}"

end_latex_document = "\\end{document}"

begin_block = re.compile(r'\\begin{(?P<block_name>[a-zA-Z0-9]+)}')

entire_block = re.compile(r'^\\begin{(?P<block>[a-zA-Z0-9]+)}.*\\end{(?P=block)}$', re.DOTALL)