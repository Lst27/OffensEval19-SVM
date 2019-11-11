from sklearn import svm
from sklearn.metrics import f1_score, precision_score, recall_score
import invertedIndex
import argparse
import sys
import re


class ModelA:

    def __init__(self, input_file, v):
        self.model = svm.SVC(C=2.0, class_weight={1: 1.8, -1: 1})
        self.vecstuff = invertedIndex.VecStuff(input_file)
        self.vecstuff.load(input_file)
        self.vec = v

    def get_vec(self, i, tweet):
        if i == 1:
            return self.vecstuff.get_combined_vec(tweet)
        elif i == 2:
            return self.vecstuff.get_positioned_combined_vec(tweet)
        else:
            return self.vecstuff.get_emb_vec(tweet)

    def train(self, file):
        data = []
        labels = []
        cnt = 0
        with open(file, 'r', encoding='utf-8', errors='ignore') as input:
            for line in input:
                if cnt % 1000 == 0:
                    print("transforming data: line " + str(cnt))
                parts = line.split('\t')
                data.append(self.get_vec(self.vec, parts[1]))
                off = parts[2]
                if off == "OFF":
                    labels.append(1)
                else:
                    labels.append(-1)
                cnt += 1
        print("start training")
        self.model.fit(data, labels)

    def train_and_eval(self, file):
        data = []
        labels = []
        eval_data = []
        eval_label = []
        cnt = 0
        with open(file, 'r', encoding='utf-8', errors='ignore') as input:
            input.readline()
            for line in input:
                parts = line.split('\t')
                d = self.get_vec(self.vec, parts[1])
                if parts[2] == "OFF":
                    l = 1
                else:
                    l = -1
                if cnt % 1000 == 0:
                    print("transforming data: line " + str(cnt))
                if cnt < 10000:
                    data.append(d)
                    labels.append(l)
                elif cnt == 10000:
                    print("start training")
                    self.model.fit(data, labels)
                if cnt >= 10000:
                    eval_data.append(d)
                    eval_label.append(l)
                cnt += 1
            print("predicting")
            estimates = self.model.predict(eval_data)
            total = 0
            correct = 0
            for (est, act) in zip(estimates, eval_label):
                if est == act:
                    correct += 1
                total += 1
            print("precision: " + str(precision_score(eval_label, estimates)))
            print("recall: " + str(recall_score(eval_label, estimates)))
            print("F1: " + str(f1_score(eval_label, estimates, average='macro')))
            print("accuracy: " +str(correct/total))

    def ten_fold(self, file):
        tenth = self.vecstuff.n_docs / 10
        overallF1 = 0
        overallP = 0
        overallR = 0
        overallA = 0
        print("i\tP\tR\tF1\tAcc")
        for i in range(10):
            data = []
            labels = []
            eval_data = []
            eval_label = []
            cnt = 0
            with open(file, 'r', encoding='utf-8', errors='ignore') as input:
                input.readline()
                for line in input:
                    parts = line.split('\t')
                    d = self.get_vec(self.vec, parts[1])
                    if parts[2] == "OFF":
                        l = 1
                    else:
                        l = -1
                    if cnt < int(i * tenth):
                        data.append(d)
                        labels.append(l)
                    elif cnt <= int((i+1) * tenth) and cnt >= int(i * tenth):
                        eval_data.append(d)
                        eval_label.append(l)
                    else:
                        data.append(d)
                        labels.append(l)
                    cnt += 1
                self.model.fit(data, labels)
                estimates = self.model.predict(eval_data)
                total = 0
                correct = 0
                for (est, act) in zip(estimates, eval_label):
                    if est == act:
                        correct += 1
                    total += 1

                overallP += precision_score(eval_label, estimates, average='macro')
                overallR += recall_score(eval_label, estimates, average='macro')
                overallF1 += f1_score(eval_label, estimates, average='macro')
                overallA += correct / total
                print("{}\t{}\t{}\t{}\t{}", i, precision_score(eval_label, estimates, average='macro'),
                      recall_score(eval_label, estimates, average='macro'),
                      f1_score(eval_label, estimates, average='macro'),
                      correct / total)
        print("Overall precision, macro: " + str(overallP/10))
        print("Overall recall, macro: " + str(overallR/10))
        print("Overall F1, macro: " + str(overallF1/10))
        print("Overall accuracy: " + str(overallA/10))

    def predict(self, predict_file, outfile):
        data = []
        ids = []
        with open(predict_file, 'r', encoding='utf-8', errors='ignore') as input:
            input.readline()
            for line in input:
                parts = line.split('\t')
                d = self.get_vec(self.vec, parts[1])
                ids.append(parts[0])
                data.append(d)
            print("start predicting")
            predictions = self.model.predict(data)
            input.close()
        assert len(ids) == len(predictions)
        with open(outfile, 'w', encoding='utf-8', errors='ignore') as output:
            for (id, pred) in zip(ids, predictions):
                if pred == -1:
                    label = 'NOT'
                else:
                    label = 'OFF'
                output.write("{},{}\n".format(id, label))
        output.close()

class ModelB:

    def __init__(self, input_file, v):
        self.model = svm.SVC(C=2.0, class_weight={1: 1, -1: 4.5})
        self.vecstuff = invertedIndex.VecStuff(input_file)
        self.vec = v

    def get_vec(self, i, tweet):
        if i == 1:
            return self.vecstuff.get_combined_vec(tweet)
        elif i == 2:
            return self.vecstuff.get_positioned_combined_vec(tweet)
        else:
            return self.vecstuff.get_emb_vec(tweet)

    def train(self, file):
        data = []
        labels = []
        cnt = 0
        with open(file, 'r', encoding='utf-8', errors='ignore') as input:
            for line in input:
                if cnt % 1000 == 0:
                    print("transforming data: line " + str(cnt))
                parts = line.split('\t')
                off = parts[2]
                target = parts[3]
                #train it only on the offensive subset
                if off == "OFF":
                    data.append(self.get_vec(self.vec, parts[1]))
                    if target == "TIN":
                        labels.append(1)
                    else:
                        labels.append(-1)
                cnt += 1
        print("start training")
        self.model.fit(data, labels)

    def predict(self, predict_file, outfile):
        data = []
        ids = []
        with open(predict_file, 'r', encoding='utf-8', errors='ignore') as input:
            input.readline()
            for line in input:
                parts = line.split('\t')
                d = self.get_vec(self.vec, parts[1])
                ids.append(parts[0])
                data.append(d)
            print("start predicting")
            predictions = self.model.predict(data)
            input.close()
        assert len(ids) == len(predictions)
        with open(outfile, 'w', encoding='utf-8', errors='ignore') as output:
            for (id, pred) in zip(ids, predictions):
                if pred == -1:
                    label = 'UNT'
                else:
                    label = 'TIN'
                output.write("{},{}\n".format(id, label))
        output.close()

    def ten_fold(self, file):
        tenth = 440
        overallF1, overallP, overallR, overallA = 0, 0, 0, 0
        print("i\tP\tR\tF1\tAcc")
        for i in range(10):
            data = []
            labels = []
            eval_data = []
            eval_label = []
            cnt = 0
            with open(file, 'r', encoding='utf-8', errors='ignore') as input:
                input.readline()
                for line in input:
                    parts = line.split('\t')
                    if parts[2] == "OFF":
                        d = self.get_vec(self.vec, parts[1])
                        if parts[3] == "TIN":
                            l = 1
                        else:
                            l = -1
                        if cnt < int(i * tenth):
                            data.append(d)
                            labels.append(l)
                        elif cnt <= int((i+1) * tenth) and cnt >= int(i * tenth):
                            eval_data.append(d)
                            eval_label.append(l)
                        else:
                            data.append(d)
                            labels.append(l)
                        cnt += 1
                if len(eval_label) == 0:
                    continue
                print("start training")
                self.model.fit(data, labels)
                print("predicting")
                estimates = self.model.predict(eval_data)
                total = 0
                correct = 0
                for (est, act) in zip(estimates, eval_label):
                    if est == act:
                        correct += 1
                    total += 1

                overallP += precision_score(eval_label, estimates, average='macro')
                overallR += recall_score(eval_label, estimates, average='macro')
                overallF1 += f1_score(eval_label, estimates, average='macro')
                overallA += correct / total
                print("{}\t{}\t{}\t{}\t{}", i, precision_score(eval_label, estimates, average='macro'),
                      recall_score(eval_label, estimates, average='macro'),
                      f1_score(eval_label, estimates, average='macro'),
                      correct / total)
        print("Overall precision, macro: " + str(overallP/10))
        print("Overall recall, macro: " + str(overallR/10))
        print("Overall F1, macro: " + str(overallF1/10))
        print("Overall accuracy: " + str(overallA/10))


class ModelC:
    #for positioned
    #model = svm.SVC(C=2, kernel='rbf', decision_function_shape='ovo', class_weight={0: 1, 1: 3.1, 2: 5.3})
    #for combined

    def __init__(self, input_file, v):
        self.model = svm.SVC(C=2, kernel='rbf', decision_function_shape='ovo', class_weight={0: 1, 1: 2.4, 2: 6})
        self.vecstuff = invertedIndex.VecStuff('../training-v1/offenseval-training-v1.tsv')
        self.vec = v

    def get_vec(self, i, tweet):
        if i == 1:
            return self.vecstuff.get_combined_vec(tweet)
        elif i == 2:
            return self.vecstuff.get_positioned_combined_vec(tweet)
        else:
            return self.vecstuff.get_emb_vec(tweet)

    def train(self, file):
        data = []
        labels = []
        cnt = 0
        with open(file, 'r', encoding='utf-8', errors='ignore') as input:
            for line in input:
                if cnt % 1000 == 0:
                    print("transforming data: line " + str(cnt))
                parts = line.split('\t')
                off = parts[2]
                targeted = parts[3]
                target = parts[4].strip()
                #train it only on the offensive and targeted subset
                if off == "OFF":
                    if targeted == "TIN":
                        data.append(self.get_vec(self.vec, parts[1]))
                        if target == "IND":
                            labels.append(0)
                        elif target == "GRP":
                            labels.append(1)
                        elif target == "OTH":
                            labels.append(2)
                cnt += 1
        print("start training")
        self.model.fit(data, labels)


    def predict(self, predict_file, outfile):
        data = []
        ids = []
        with open(predict_file, 'r', encoding='utf-8', errors='ignore') as input:
            input.readline()
            for line in input:
                parts = line.split('\t')
                d = self.get_vec(self.vec, parts[1])
                ids.append(parts[0])
                data.append(d)
            print("start predicting")
            predictions = self.model.predict(data)
            input.close()
        assert len(ids) == len(predictions)
        with open(outfile, 'w', encoding='utf-8', errors='ignore') as output:
            for (id, pred) in zip(ids, predictions):
                if pred == 0:
                    label = 'IND'
                elif pred == 1:
                    label = 'GRP'
                else:
                    label = "OTH"
                output.write("{},{}\n".format(id, label))
        output.close()


    def ten_fold(self, file):
        tenth = 387
        overallF1, overallP, overallR, overallA = 0, 0, 0, 0
        print("i\tP\tR\tF1\tAcc")
        for i in range(10):
            data = []
            labels = []
            eval_data = []
            eval_label = []
            cnt = 0
            with open(file, 'r', encoding='utf-8', errors='ignore') as input:
                input.readline()
                for line in input:
                    parts = line.split('\t')
                    if parts[2].strip() == "OFF":
                        if parts[3].strip() == "TIN":
                            d = self.get_vec(self.vec, parts[1])
                            if parts[4].strip() == "IND":
                                l = 0
                            elif parts[4].strip() == "GRP":
                                l = 1
                            elif parts[4].strip() == "OTH":
                                l = 2
                            if cnt < int(i * tenth):
                                data.append(d)
                                labels.append(l)
                            elif cnt <= int((i+1) * tenth) and cnt >= int(i * tenth):
                                eval_data.append(d)
                                eval_label.append(l)
                            else:
                                data.append(d)
                                labels.append(l)
                            cnt += 1
                if len(eval_label) == 0:
                    continue
                print("start training")
                self.model.fit(data, labels)
                print("predicting")
                estimates = self.model.predict(eval_data)
                total = 0
                correct = 0
                zeros, ones, twos, actzeros, actones, acttwos = 0,0,0,0,0,0
                for (est, act) in zip(estimates, eval_label):
                    if est == act:
                        correct += 1
                    total += 1
                    if est == 0:
                        zeros +=1
                    elif est == 1:
                        ones +=1
                    else:
                        twos += 1
                    if act == 0:
                        actzeros += 1
                    elif act == 1:
                        actones += 1
                    else:
                        acttwos += 1
                overallP += precision_score(eval_label, estimates, average='macro')
                overallR += recall_score(eval_label, estimates, average='macro')
                overallF1 += f1_score(eval_label, estimates, average='macro')
                overallA += correct / total
                print("{}\t{}\t{}\t{}\t{}", i, precision_score(eval_label, estimates, average='macro'),
                      recall_score(eval_label, estimates, average='macro'),
                      f1_score(eval_label, estimates, average='macro'),
                      correct / total)
        print("Overall precision, macro: " + str(overallP/10))
        print("Overall recall, macro: " + str(overallR/10))
        print("Overall F1, macro: " + str(overallF1/10))
        print("Overall accuracy: " + str(overallA/10))


def model_type(c):
    valid = re.compile('^[A|B|C]$')
    if not valid.match(c):
        msg = "%r is not a valid Model type" % c
        raise argparse.ArgumentTypeError(msg)
    return c


def get_model(m, input_file, vector_type):
    if m == 'A':
        return ModelA(input_file, vector_type)
    elif m == 'B':
        return ModelB(input_file, vector_type)
    else:
        return ModelC(input_file, vector_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SVM',
                                     epilog="Place a binary fastext model file (../cc.en.300.bin) and vader lexicon (../vader.txt) in the "
                                            "parent directory of the python project folder")
    parser.add_argument('--vector', '-v', type=int, choices=range(1, 4), nargs='?',
                        help="Choose vector representation used in the model: \n "
                             "1: combined_fixed \n 2: combined_positioned \n 3: emb_only \n default is combined_fixed",
                        default=1)
    parser.add_argument('--model', '-m', metavar='M', type=model_type, nargs='?',
                        help="Choose subtask specific model: A, B, C \n default is A", default='A')
    parser.add_argument('infile_training', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('infile_prediction', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('outfile', type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument('--tenfold', '-t', action="store_true",
                        help="Perform tenfold validation and print evaluation metrics, does not print predictions to file")
    args = parser.parse_args()
    infile = args.infile_training.name
    infile_predict = args.infile_prediction.name
    outfile = args.outfile.name
    v = args.vector
    m = args.model

    model = get_model(m, infile, v)
    if args.tenfold:
        model.ten_fold(infile)
    else:
        model.train(infile)
        model.predict(infile_predict, outfile)

