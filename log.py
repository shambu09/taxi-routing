log_file = """
------------------Log File-------------------

(20, 4)
(20, 1)


centers
[[ 38.         309.5        211.5          9.           0.
    1.        ]
 [192.25        11.75        86.         242.           1.
    1.        ]
 [ 86.5        307.         255.         248.5          2.
    1.        ]
 [292.33333333 257.16666667 292.33333333 257.16666667   3.
    1.        ]
 [ 51.875      143.25        51.875      143.25         4.
    1.        ]
 [ 37.          84.33333333 243.33333333 314.66666667   5.
    1.        ]
 [351.6        203.6        187.2         51.2          6.
    1.        ]
 [228.66666667 289.          62.66666667 256.66666667   7.
    1.        ]
 [218.75        49.75       357.5        180.75         8.
    1.        ]
 [164.66666667 209.33333333 127.          23.           9.
    1.        ]]
(10, 6)
"""





























































#####################################
################CODE#################
#####################################
class Logger:
    __init = "\n------------------Log File-------------------\n"

    def __writeFile(func):
        def inner(value=None):
            string, mode = func(str(value))
            with open(__file__, mode) as f:
                f.write(string)
        return inner

    def __readFile(func):
        def inner(*args):
            with open(__file__, "r") as f:
                string, i = func(f.read(), *args)
            return string, i
        return inner

    @__readFile
    def __preprocess(string, log_string):
        i = -1*string[::-1].find('\n"""', 800)-5
        if log_string=="None":
            log_string = "\n"
        else:
            log_string = "\n" + log_string
        new_code = string[0:14] + string[14:i] + log_string + string[i: ]
        return new_code, i

    @__writeFile
    def log(string):
        new_code, _ = Logger.__preprocess(string)
        return new_code, "w"

    @__writeFile
    def clear(string):
        code, i = Logger.__preprocess(Logger.__init)
        new_code = code[0:14] + Logger.__init + code[i: ]
        return new_code, "w"
