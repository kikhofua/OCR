# TODO: I need to create a regex that searches for latex tokens to ignore
# TODO: I need to create { token => id } and [token,...]

import re


begin_latex_document = "\\begin{document}"

end_latex_document = "\\end{document}"

begin_block = re.compile(r"""
    \\begin{(?P<block_name>[a-zA-Z0-9]+)}
    | \$\$
""", re.VERBOSE)

entire_block = re.compile(r"""
    ^\\begin{(?P<block>[a-zA-Z0-9]+)}.*\\end{(?P=block)}$
    | \$\$.*\$\$
""", re.DOTALL | re.VERBOSE)

bad_latex_tokens_regex = re.compile(r"""
    \\([a-zA-Z0-9])*cite([a-zA-Z0-9])*{.*}
    | \\footnote(\[.*\])?{.*}
    | \\\\\*
    | \\kill
    | \\pagebreak
    | \\today
    | \\[hv]space{.*}
    | \\rule{.*}{.*}
    | \\(label|ref|pageref|footnote){\S*}
    | \\begin{(table|figure)}.*\\end{(table|figure)}
    | \\caption{.*}
""", re.VERBOSE)

inline_math_regex = re.compile(r"""
    \$                      # start inline math expression
    (?P<math_content>.*)    # the content of math mode
    \$                      # end inline math expression
""", re.VERBOSE | re.DOTALL)

math_block_regex = re.compile(r"""
    (\\begin{(equation|multiline|align|eqnarray|IEEEeqnarray|cases)} | \$\$ | \$)       # starting math mode
    (?P<math_content>.*)  # the content of math mode
    (\\end{(equation|multiline|align|eqnarray|IEEEeqnarray|cases)} | \$\$ | \$)         # ending math mode
""", re.VERBOSE | re.DOTALL)

bibliography_regex = re.compile(r"""
    \\begin{thebibliography}
    .*
    \\end{thebibliography}
""", re.VERBOSE | re.DOTALL)

valid_math_token = re.compile(r"""
    \\longleftrightarrow
    | \\Longleftrightarrow
    | \\scriptscriptystyle
    | \\bigtriangledown
    | \\langle\\!\\langle
    | \\rangle\\!\\rangle
    | \\leftrightarrow
    | \\Leftrightarrow
    | \\longrightarrow
    | \\Longrightarrow
    | \\hookrightarrow
    | \\subsubsection
    | \\textbackslash
    | \\triangleright
    | \\bigtriangleup
    | \\hookleftarrow
    | \\longleftarrow
    | \\Longleftarrow
    | \\displayindent
    | \\subparagraph
    | \\footnotesize
    | \\triangleleft
    | \\displaystyle
    | \\displaylines
    | \\begin{IEEEeqnarray}
    | \\end{IEEEeqnarray}
    | \\raggedright
    | \\textgreater
    | \\diamondsuit
    | \\updownarrow
    | \\Updownarrow
    | \\scriptstyle
    | \\subsection
    | \\begin{description}
    | \\end{description}
    | \\textnormal
    | \\normalfont
    | \\scriptsize
    | \\normalsize
    | \\verb!text!
    | \\raggedleft
    | \\textbullet
    | \\sqsubseteq
    | \\sqsupseteq
    | \\rightarrow
    | \\Rightarrow
    | \\longmapsto
    | \\hangindent
    | \\legalignno
    | \\varepsilon
    | \\paragraph
    | \\begin{flushright}
    | \\end{flushright}
    | \\begin{array}
    | \\end{array}
    | \\begin{cases}
    | \\end{cases}
    | \\centering
    | \\backslash
    | \\heartsuit
    | \\spadesuit
    | \\bigotimes
    | \\leftarrow
    | \\Leftarrow
    | \\downarrow
    | \\Downarrow
    | \\widetilde
    | \\underline
    | \\textstyle
    | \\parindent
    | \\rightskip
    | \\hangafter
    | \\eqalignno
    | \\begin{quotation}
    | \\end{quotation}
    | \\begin{enumerate}
    | \\end{enumerate}
    | \\begin{flushleft}
    | \\end{flushleft}
    | \\begin{multiline}
    | \\end{multiline}
    | \\textless
    | \\varsigma
    | \\emptyset
    | \\triangle
    | \\clubsuit
    | \\bigsqcup
    | \\bigwedge
    | \\bigoplus
    | \\biguplus
    | \\subseteq
    | \\supseteq
    | \\parallet
    | \\buildrel
    | \\overline
    | \\underbar
    | \\noindent
    | \\leftskip
    | \\narrower
    | \\itemitem
    | \\vartheta
    | \\chapter
    | \\section
    | \\begin{verbatim}
    | \\end{verbatim}
    | \\begin{equation}
    | \\end{equation}
    | \\begin{eqnarray}
    | \\end{eqnarray}
    | \\textbar
    | \$\\sim\$
    | \\partial
    | \\natural
    | \\bigodot
    | \\diamond
    | \\bigcirc
    | \\ddagger
    | \\uparrow
    | \\nearrow
    | \\nwarrow
    | \\Uparrow
    | \\searrow
    | \\swarrow
    | \\widehat
    | \\eqalign
    | \\noalign
    | \\epsilon
    | \\upsilon
    | \\Upsilon
    | \\begin{comment}
    | \\end{comment}
    | \\begin{itemize}
    | \\end{itemize}
    | \\textrm
    | \\textsf
    | \\texttt
    | \\textmd
    | \\textbf
    | \\textup
    | \\textit
    | \\textsl
    | \\textsc
    | \\varrho
    | \\varphi
    | \\forall
    | \\exists
    | \\coprod
    | \\bigcap
    | \\bigcup
    | \\bigvee
    | \\bullet
    | \\ominus
    | \\otimes
    | \\oslash
    | \\dagger
    | \\preceq
    | \\subset
    | \\propto
    | \\succeq
    | \\supset
    | \\approx
    | \\bowtie
    | \\models
    | \\mapsto
    | \\lbrack
    | \\rbrack
    | \\lbrace
    | \\rbrace
    | \\lfloor
    | \\rfloor
    | \(\\!\(
    | \)\\!\)
    | \\langle
    | \\rangle
    | \\choose
    | \\arccos
    | \\arcsin
    | \\arctan
    | \\liminf
    | \\limsup
    | \\indent
    | \\lambda
    | \\Lambda
    | \\begin{center}
    | \\end{center}
    | \\small
    | \\large
    | \\Large
    | \\varpi
    | \\aleph
    | \\prime
    | \\imath
    | \\jmath
    | \\infty
    | \\nabla
    | \\angle
    | \\sharp
    | \\uplus
    | \\sqcap
    | \\sqcup
    | \\wedge
    | \\oplus
    | \\amalg
    | \\vdash
    | \\smile
    | \\frown
    | \\dashv
    | \\equiv
    | \\simeq
    | \\asymp
    | \\doteq
    | \[\\!\[
    | \]\\!\]
    | \\lceil
    | \\rceil
    | \\right
    | \\biggl
    | \\biggr
    | \\check
    | \\tilde
    | \\acute
    | \\grave
    | \\breve
    | \\brace
    | \\brack
    | \\ldots
    | \\vdots
    | \\ddots
    | \\leqno
    | \\cases
    | \\alpha
    | \\delta
    | \\sigma
    | \\omega
    | \\Theta
    | \\times
    | \\Gamma
    | \\Sigma
    | \\cdots
    | \\gamma
    | \\theta
    | \\kappa
    | \\Delta
    | \\Omega
    | \\part
    | \\begin{quote}
    | \\end{quote}
    | \\begin{verse}
    | \\end{verse}
    | \\begin{align}
    | \\end{align}
    | \\item
    | \\emph
    | \\tiny
    | \\huge
    | \\Huge
    | \\\^{}
    | \\t oo
    | \\hbar
    | \\surd
    | \\lnot
    | \\prod
    | \\oint
    | \\star
    | \\land
    | \\odot
    | \\prec
    | \\succ
    | \\cong
    | \\owns
    | \\perp
    | \\gets
    | \\over
    | \\vert
    | \\Vert
    | \\left
    | \\bigl
    | \\bigr
    | \\Bigl
    | \\Bigr
    | \\bigm
    | \\ddot
    | \\root
    | \\atop
    | \\pmod
    | \\bmod
    | \\cosh
    | \\coth
    | \\sinh
    | \\tanh
    | \\dots
    | \\eqno
    | \\frac
    | \\sqrt
    | \\cdot
    | \\zeta
    | \\circ
    | \\beta
    | \\iota
    | \\~{}
    | \\c c
    | \\d o
    | \\\^o
    | \\c o
    | \\b o
    | \\v o
    | \\H o
    | \\ell
    | \\top
    | \\bot
    | \\neg
    | \\int
    | \\cap
    | \\cup
    | \\vee
    | \\lor
    | \\not
    | \\mid
    | \\sim
    | \\big
    | \\hat
    | \\dot
    | \\bar
    | \\vec
    | \\arg
    | \\cos
    | \\cot
    | \\csc
    | \\deg
    | \\det
    | \\dim
    | \\exp
    | \\gcd
    | \\hom
    | \\inf
    | \\ker
    | \\lim
    | \\log
    | \\max
    | \\min
    | \\sec
    | \\sin
    | \\sup
    | \\tan
    | \\sum
    | \\leq
    | \\ast
    | \\phi
    | \\Phi
    | \\geq
    | \\eta
    | \\tau
    | \\chi
    | \\Psi
    | \\neq
    | \\div
    | \\rho
    | \\psi
    | \\rm
    | \\sf
    | \\tt
    | \\md
    | \\bf
    | \\up
    | \\it
    | \\sl
    | \\sc
    | \\em
    | \\\\$
    | \\\#
    | \\‘o
    | \\\\.
    | \\OE
    | \\’o
    | \\"o
    | \\ae
    | \\AE
    | \\?‘
    | \\~o
    | \\aa
    | \\=o
    | \\oe
    | \\AA
    | ---
    | \\wp
    | \\ne
    | \\Re
    | \\Im
    | \\\|
    | \\pm
    | \\mp
    | \\wr
    | \\le
    | \\ll
    | \\in
    | \\ge
    | \\gg
    | \\ni
    | \\to
    | \\of
    | \\lg
    | \\ln
    | \\Pr
    | \\xi
    | \\Pi
    | \\mu
    | \\pi
    | \\nu
    | \\Xi
    | \$\$
    | \\&
    | \\%
    | \\_
    | \\o
    | \\j
    | \\O
    | ~‘
    | \\l
    | \\L
    | \\i
    | ''
    | ``
    | --
    | \\\\
    | \\@
    | \\{
    | \\}
    | \\,
    | \\ #
    | [0-9a-zA-Z!"#%&'()<>*+,\-./:;?$@[\\\]^_`{|}~=]   # non-latex tokens (ascii)
""", re.VERBOSE)
