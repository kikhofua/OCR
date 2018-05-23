# import os, re
# from data.generation.utils import bad_latex_tokens_regex, end_latex_document, begin_block, entire_block, math_block_regex, bibliography_regex
# script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
# rel_dir = "expanded_latex"
# abs_file_dir = os.path.join(script_dir, rel_dir)
#
# token_regex = re.compile(r"""\\[a-zA-Z]+""", re.VERBOSE)
# valid_math_token = re.compile(r"""
#     ^
#     \\longleftrightarrow
#     | \\Longleftrightarrow
#     | \\scriptscriptystyle
#     | \\bigtriangledown
#     | \\langle\\!\\langle
#     | \\rangle\\!\\rangle
#     | \\leftrightarrow
#     | \\Leftrightarrow
#     | \\longrightarrow
#     | \\Longrightarrow
#     | \\hookrightarrow
#     | \\subsubsection
#     | \\textbackslash
#     | \\triangleright
#     | \\bigtriangleup
#     | \\hookleftarrow
#     | \\longleftarrow
#     | \\Longleftarrow
#     | \\displayindent
#     | \\subparagraph
#     | \\footnotesize
#     | \\triangleleft
#     | \\displaystyle
#     | \\displaylines
#     | \\begin{IEEEeqnarray}
#     | \\end{IEEEeqnarray}
#     | \\raggedright
#     | \\textgreater
#     | \\diamondsuit
#     | \\updownarrow
#     | \\Updownarrow
#     | \\scriptstyle
#     | \\subsection
#     | \\begin{description}
#     | \\end{description}
#     | \\textnormal
#     | \\normalfont
#     | \\scriptsize
#     | \\normalsize
#     | \\verb!text!
#     | \\raggedleft
#     | \\textbullet
#     | \\sqsubseteq
#     | \\sqsupseteq
#     | \\rightarrow
#     | \\Rightarrow
#     | \\longmapsto
#     | \\hangindent
#     | \\legalignno
#     | \\varepsilon
#     | \\paragraph
#     | \\begin{flushright}
#     | \\end{flushright}
#     | \\begin{array}
#     | \\end{array}
#     | \\begin{cases}
#     | \\end{cases}
#     | \\centering
#     | \\backslash
#     | \\heartsuit
#     | \\spadesuit
#     | \\bigotimes
#     | \\leftarrow
#     | \\Leftarrow
#     | \\downarrow
#     | \\Downarrow
#     | \\widetilde
#     | \\underline
#     | \\textstyle
#     | \\parindent
#     | \\rightskip
#     | \\hangafter
#     | \\eqalignno
#     | \\begin{quotation}
#     | \\end{quotation}
#     | \\begin{enumerate}
#     | \\end{enumerate}
#     | \\begin{flushleft}
#     | \\end{flushleft}
#     | \\begin{multiline}
#     | \\end{multiline}
#     | \\textless
#     | \\varsigma
#     | \\emptyset
#     | \\triangle
#     | \\clubsuit
#     | \\bigsqcup
#     | \\bigwedge
#     | \\bigoplus
#     | \\biguplus
#     | \\subseteq
#     | \\supseteq
#     | \\parallet
#     | \\buildrel
#     | \\overline
#     | \\underbar
#     | \\noindent
#     | \\leftskip
#     | \\narrower
#     | \\itemitem
#     | \\vartheta
#     | \\chapter
#     | \\section
#     | \\begin{verbatim}
#     | \\end{verbatim}
#     | \\begin{equation}
#     | \\end{equation}
#     | \\begin{eqnarray}
#     | \\end{eqnarray}
#     | \\textbar
#     | \$\\sim\$
#     | \\partial
#     | \\natural
#     | \\bigodot
#     | \\diamond
#     | \\bigcirc
#     | \\ddagger
#     | \\uparrow
#     | \\nearrow
#     | \\nwarrow
#     | \\Uparrow
#     | \\nonumber
#     | \\searrow
#     | \\swarrow
#     | \\widehat
#     | \\eqalign
#     | \\noalign
#     | \\epsilon
#     | \\upsilon
#     | \\Upsilon
#     | \\begin{comment}
#     | \\end{comment}
#     | \\begin{itemize}
#     | \\end{itemize}
#     | \\textrm
#     | \\textsf
#     | \\texttt
#     | \\textmd
#     | \\textbf
#     | \\textup
#     | \\textit
#     | \\textsl
#     | \\textsc
#     | \\varrho
#     | \\varphi
#     | \\forall
#     | \\exists
#     | \\coprod
#     | \\bigcap
#     | \\bigcup
#     | \\bigvee
#     | \\bullet
#     | \\ominus
#     | \\otimes
#     | \\oslash
#     | \\dagger
#     | \\preceq
#     | \\subset
#     | \\limits
#     | \\propto
#     | \\succeq
#     | \\supset
#     | \\approx
#     | \\bowtie
#     | \\models
#     | \\mapsto
#     | \\lbrack
#     | \\rbrack
#     | \\lbrace
#     | \\rbrace
#     | \\lfloor
#     | \\rfloor
#     | \(\\!\(
#     | \)\\!\)
#     | \\langle
#     | \\rangle
#     | \\choose
#     | \\arccos
#     | \\arcsin
#     | \\arctan
#     | \\liminf
#     | \\limsup
#     | \\indent
#     | \\lambda
#     | \\Lambda
#     | \\begin{center}
#     | \\end{center}
#     | \\small
#     | \\large
#     | \\Large
#     | \\varpi
#     | \\aleph
#     | \\prime
#     | \\imath
#     | \\jmath
#     | \\infty
#     | \\nabla
#     | \\angle
#     | \\sharp
#     | \\uplus
#     | \\sqcap
#     | \\sqcup
#     | \\wedge
#     | \\oplus
#     | \\qquad
#     | \\amalg
#     | \\vdash
#     | \\smile
#     | \\frown
#     | \\dashv
#     | \\equiv
#     | \\simeq
#     | \\asymp
#     | \\doteq
#     | \[\\!\[
#     | \]\\!\]
#     | \\lceil
#     | \\rceil
#     | \\right
#     | \\biggl
#     | \\biggr
#     | \\check
#     | \\tilde
#     | \\acute
#     | \\grave
#     | \\breve
#     | \\brace
#     | \\brack
#     | \\ldots
#     | \\vdots
#     | \\ddots
#     | \\leqno
#     | \\cases
#     | \\alpha
#     | \\delta
#     | \\sigma
#     | \\omega
#     | \\Theta
#     | \\times
#     | \\Gamma
#     | \\Sigma
#     | \\cdots
#     | \\gamma
#     | \\theta
#     | \\hbox{\S*}
#     | \\mbox{\S*}
#     | \\kappa
#     | \\Delta
#     | \\Omega
#     | \\part
#     | \\begin{quote}
#     | \\end{quote}
#     | \\begin{verse}
#     | \\end{verse}
#     | \\begin{align}
#     | \\end{align}
#     | \\item
#     | \\emph
#     | \\tiny
#     | \\huge
#     | \\Huge
#     | \\\^{}
#     | \\t oo
#     | \\hbar
#     | \\hbox
#     | \\surd
#     | \\lnot
#     | \\prod
#     | \\oint
#     | \\quad
#     | \\star
#     | \\land
#     | \\odot
#     | \\prec
#     | \\succ
#     | \\cong
#     | \\owns
#     | \\perp
#     | \\gets
#     | \\over
#     | \\vert
#     | \\Vert
#     | \\left
#     | \\bigl
#     | \\bigr
#     | \\Bigl
#     | \\Bigr
#     | \\bigm
#     | \\ddot
#     | \\root
#     | \\atop
#     | \\pmod
#     | \\bmod
#     | \\cosh
#     | \\coth
#     | \\sinh
#     | \\tanh
#     | \\dots
#     | \\eqno
#     | \\frac
#     | \\sqrt
#     | \\cdot
#     | \\zeta
#     | \\circ
#     | \\beta
#     | \\iota
#     | \\~{}
#     | \\c c
#     | \\d o
#     | \\\^o
#     | \\c o
#     | \\b o
#     | \\v o
#     | \\H o
#     | \\ell
#     | \\top
#     | \\bot
#     | \\neg
#     | \\int
#     | \\cap
#     | \\cup
#     | \\vee
#     | \\lor
#     | \\not
#     | \\mid
#     | \\sim
#     | \\big
#     | \\Big
#     | \\cal
#     | \\hat
#     | \\dot
#     | \\bar
#     | \\vec
#     | \\arg
#     | \\cos
#     | \\cot
#     | \\csc
#     | \\deg
#     | \\det
#     | \\dim
#     | \\exp
#     | \\gcd
#     | \\hom
#     | \\inf
#     | \\ker
#     | \\lim
#     | \\log
#     | \\max
#     | \\min
#     | \\sec
#     | \\sin
#     | \\sup
#     | \\tan
#     | \\sum
#     | \\leq
#     | \\ast
#     | \\phi
#     | \\Phi
#     | \\geq
#     | \\eta
#     | \\tau
#     | \\chi
#     | \\Psi
#     | \\neq
#     | \\div
#     | \\rho
#     | \\psi
#     | \\rm
#     | \\sf
#     | \\tt
#     | \\md
#     | \\bf
#     | \\up
#     | \\it
#     | \\sl
#     | \\sc
#     | \\em
#     | \\\\$
#     | \\\#
#     | \\‘o
#     | \\\\.
#     | \\OE
#     | \\’o
#     | \\"o
#     | \\ae
#     | \\AE
#     | \\?‘
#     | \\~o
#     | \\aa
#     | \\=o
#     | \\oe
#     | \\AA
#     | ---
#     | \\wp
#     | \\ne
#     | \\Re
#     | \\Im
#     | \\\|
#     | \\pm
#     | \\mp
#     | \\wr
#     | \\le
#     | \\ll
#     | \\in
#     | \\ge
#     | \\gg
#     | \\ni
#     | \\to
#     | \\of
#     | \\lg
#     | \\ln
#     | \\cr
#     | \\Pr
#     | \\xi
#     | \\Pi
#     | \\mu
#     | \\pi
#     | \\nu
#     | \\Xi
#     | \$\$
#     | \\&
#     | \\%
#     | \\_
#     | \\o
#     | \\j
#     | \\O
#     | ~‘
#     | \\l
#     | \\L
#     | \\i
#     | ''
#     | ``
#     | --
#     | \\\\
#     | \\@
#     | \\{
#     | \\}
#     | \\,
#     | \\ # white space
#     $
# """, re.VERBOSE)
# begin_latex_document = "\\begin{document}"
# # def_regex = re.compile(r"""\\def\s*(?P<custom_macro>\\[a-zA-Z]+)""")
# # newcommand_regex = re.compile(r"""\\newcommand{(?P<custom_macro>\\[a-zA-Z]+)}""")
#
# all_tokens = {}
# count = 0
# for doc in os.listdir(abs_file_dir):
#     file = os.path.join(abs_file_dir, doc)
#     in_document_body = False
#     building_block = False
#     building_inline = False
#     multi_line_builder = ""
#     with open(file, 'r') as document:
#         for line in document:
#             if not in_document_body:
#                 if begin_latex_document in line:
#                     in_document_body = True
#             else:
#                 line = line.strip()
#                 # line = re.sub(bad_latex_tokens_regex, "", line)
#                 if not line or line.startswith("%"):
#                     continue
#                 if end_latex_document in line:
#                     break
#                 if not building_block:
#                     matches_begin_block = re.search(begin_block, line)
#                     if matches_begin_block:
#                         building_block = True
#                     elif "\\" not in line or "$" not in line:  # skip the line if there's nothing interesting
#                         continue
#                     elif line.count("$") % 2 == 1:
#                         building_inline = True
#                 if building_block:
#                     new_block = multi_line_builder + line
#                     match = re.search(entire_block, new_block)
#                     if match:
#                         multi_line_builder = ""
#                         building_block = False
#                         math_block = re.search(math_block_regex, new_block)
#                         bibliography = re.search(bibliography_regex, new_block)
#                         if bibliography:
#                             line = ""
#                         elif math_block:
#                             # line = self._sparsify_math_blocks(new_block)
#                             pass
#                         else:
#                             line = new_block
#                     else:
#                         multi_line_builder += "{} ".format(line)
#                         continue
#                 elif building_inline:
#                     new_block = multi_line_builder + line
#                     if new_block.count("$") % 2 == 1:
#                         multi_line_builder += "{} ".format(line)
#                         continue
#                     else:
#                         multi_line_builder = ""
#                         building_inline = False
#                         line = new_block
#                 found_tokens = re.findall(token_regex, line)
#                 for t in found_tokens:
#                     match = re.search(valid_math_token, t)
#                     if match is None:
#                         all_tokens[t] = line
#                     # match = re.search(valid_math_token, t)
#                     # if match is None:
#                     #     all_tokens[t] = line
#     # if count > 50:
#     #     break
#     print(count); count += 1
#
#
# print("###################################################")
# with open("missing tokens 2.txt", 'w') as f:
#     for k, v in all_tokens.items():
#         # f.write("{}\t\t\t\t\t\t{}".format(k, v))
#         token_index = v.index(k)
#         start = max(0, token_index - 20)
#         end = min(len(v), token_index + len(k) + 20)
#         surroundings = v[start: end]
#         f.write("{}\t\t\t\t\t\t{}\n".format(k, surroundings))

content = []
with open("missing tokens 2.txt", 'r') as f:
    for line in f:
        line = line.strip()
        parts = line.split("\t\t\t\t\t\t")
        content.append((parts[0], parts[1]))
content = sorted(content, key=lambda x: len(x[0]), reverse=True)
for parts in content:
    print("    | \{}(".format(parts[0]))
