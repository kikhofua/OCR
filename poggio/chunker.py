from itertools import islice

pre = "\\documentclass[12pt]{article}" + "\n" + "\\begin{document}"

post = "\\end{document}"

def chunkify():
    num = 0
    for line in open("test.tex"):
        if line != "\documentclass[12pt]{article}" or line != "\\begin{document}" or line != "\\end{document}" \
                or not line.strip():
            with open(str(num) + ".tex", "w+") as s:
                s.write(pre)
                s.write("\n")
                s.write(line)
                s.write("\n")
                s.write(post)
                num += 1
            pass
        return

if __name__ == "__main__":
    chunkify()



#
#
# def make_chunk():
#     num = 0
#     with open("test.tex", "r") as f:
#         try:
#             chunk = "start"
#             while len(list(chunk)) > 0:
#                 chunk = islice(f, 5)
#                 with open(str(num), "w+") as s:
#                     s.write(pre)
#                     for line in chunk:
#                         s.write(line)
#                     s.write(post)
#                 num += 1
#         except EOFError:
#             pass
#     return 1
#
# if __name__ == "__main__":
#     make_chunk()



