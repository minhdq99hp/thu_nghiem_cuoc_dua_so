import timeit

class Tester:
    @staticmethod
    def run_and_analyze(my_function, *args):
        start = timeit.default_timer()
        print(*args)
        output = my_function(*args)
        stop = timeit.default_timer()

        print("Function \"{name}()\" run in the interval of {interval}s".format(name=my_function.__name__, interval=format(stop-start, '.2f')))
        print("Return: {type} - {value}".format(type=type(output), value=output))

        return output

    def print_interval(self, start_time, stop_time):
        print("Interval: " + str(format(stop_time-start_time, '.6f')))

def hello(a, b):
    return a + b


if __name__ == '__main__':
    tester = Tester()

    tester.run_and_analyze(hello, 12, 3)
