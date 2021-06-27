log_file = """
------------------Log File-------------------

(20, 4)
(20, 1)
[[ 43. 311.  96.   4.]
 [ 33. 308. 327.  14.]
 [142.   3. 158. 188.]
 [204.  26. 110. 161.]
 [112. 272. 254. 290.]
 [314. 167. 276. 199.]
 [333. 348. 288. 234.]
 [263. 362. 280. 233.]
 [ 73. 145. 111. 183.]
 [ 14. 146.  58.  50.]
 [ 41. 153.  76. 129.]
 [ 42. 183.   0. 157.]
 [ 15. 156. 176. 370.]
 [ 61.  57. 328. 295.]
 [397. 137. 225.  52.]
 [256. 207.  61. 342.]
 [131.  84. 316. 169.]
 [277.  46. 378. 330.]
 [242.  17. 339.  87.]
 [226. 279.  35.  40.]]
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
