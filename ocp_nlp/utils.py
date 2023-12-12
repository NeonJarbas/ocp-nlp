import concurrent.futures
import random

import time


class ParallelWorkers:
    workers = 12

    def do_work(self, arg_list, match_func):
        results = {}

        # do the work in parallel instead of sequentially
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:

            matchers = {}
            # create a unique wrapper for each worker with their arguments
            for args in arg_list:
                def do_thing(u=args):
                    return match_func(u)
                matchers[args] = do_thing

            # Start the operations and mark each future with its source
            future_to_source = {
                executor.submit(matchers[s]): s
                for s in arg_list
            }

            # retrieve results as they come
            for future in concurrent.futures.as_completed(future_to_source):
                utt = future_to_source[future]
                results[utt] = future.result()

        # all work done!
        return results


if __name__ == "__main__":
    # each item in list is the args to be passed
    # must be hashable (no dicts or lists)
    utts = ["A", "B", "C"]

    def heavy_work(u):  # arg from list above
        print("I'm doing it", u)
        time.sleep(random.randint(1, 3))
        r = random.choice([0, 1])
        print("##", u, r)
        return r


    t = ParallelWorkers()
    res = t.do_work(utts, heavy_work)
    print(res)  # {1.2: 0, False: 1, 'A': 1, None: 0}