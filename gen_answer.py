from sys import argv

idx_file = argv[1]
test_file = argv[2]
out_file = argv[3]
with_answer = False
if len(argv) > 4:
    with_answer = True

def parse_stories(lines, with_answer=True):
    stories = []
    for line in lines:
        line = line.strip()
        if line:
            _, line = line.split(' ', 1)
            if line:
                if '\t' in line:  # query line
                    answer = ''
                    if with_answer:
                        _, answer, _, candidates = line.split('\t')
                        answer = answer
                    else:
                        _, _, candidates = line.split('\t')

                    # use the first 10
                    candidates = candidates.split('|')[:10]
                    stories.append((answer, candidates))
    return stories

def main():
    with open(idx_file, "r") as idf:
        indices = [int(idx.strip()) for idx in idf.readlines()]

    with open(test_file, "r") as tf:
        answers, candidates = zip(*parse_stories(tf.readlines(), with_answer=with_answer))


    gen_answers = [cands[idx] for idx, cands in zip(indices, candidates)]

    if with_answer:
        correct = sum([1 if ga == a else 0 for ga, a in zip(gen_answers, answers)])
        print("ACC: {:.2%}".format(correct / len(answers)))

    with open(out_file, "w") as out:
        print('\n'.join(gen_answers), file=out)

if __name__ == '__main__':
    main()