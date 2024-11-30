from lm import LanguageModel
from corpus import tokenize, detokenize

# standard way to present 'main' function:
if __name__ == '__main__':
    with open('C:\\Users\\vgoum\\PycharmProjects\\ProgrammingCL\\train_shakespeare.txt') as f:
        text = f.readlines()

    tokens = [tokenize(line) for line in text]

    print('Hi! This is the program for generating a text.')
    print('You will be suggested to enter n, which defines an n-gram')
    print('N-gram is basically a sequence of n consecutive words in a text')
    print('We will build a language model based on probability distribution which follow an n-gram')
    ask_for_ngrams = input('Print "yes" if you want to try ')
    if ask_for_ngrams.lower() == 'yes':
        n = int(input('Enter n for the n-grams: '))
        lm = LanguageModel(n + 1)
        lm.train(tokens)
        print(f'Here is a text that was generated with {n}-grams:')
        print(detokenize(lm.generate()))
        texts_quantity = int(input('How many more texts would you like to generate? '))
        if texts_quantity == 0:
            print('Okay, thanks for using this program!')
        else:
            with open('C:\\Users\\vgoum\\PycharmProjects\\ProgrammingCL\\new_shakespeare.txt', 'w') as f:
                for sentence in range(texts_quantity):
                    more_generated_texts = detokenize(lm.generate())
                    f.write(more_generated_texts + '\n======================\n')

            print(f'{texts_quantity} texts were generated for you. Check it out in "new_shakespeare.txt" file!')
