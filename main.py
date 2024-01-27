#######################################################
# Modules
#######################################################
from lib.similarity_component import get_most_similar_question_and_answer

#######################################################
# Imports
#######################################################
import wikipedia
import aiml

#######################################################
# Init aiml
#######################################################
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="learn-files/mybot_basic.xml")


#######################################################
# Main executable
#######################################################
def main():
    """
    Main executable for the chat-bot project.
    """
    print("Welcome to this chat bot. Please feel free to ask questions from me!")

    while True:
        # get user input
        try:
            userInput = input("> ")
        except (KeyboardInterrupt, EOFError) as e:
            print("Bye!")
            break

        # pre-process user input and determine response agent (if needed)
        responseAgent = 'aiml'
        # activate selected response agent
        if responseAgent == 'aiml':
            answer = kern.respond(userInput)

        # post-process the answer for commands
        if answer[0] == '#':
            params = answer[1:].split('$')
            print(params[0])
            cmd = int(params[0])
            if cmd == 0:
                print(params[1])
                break
            elif cmd == 99:
                # Check question and answer bank
                response = get_most_similar_question_and_answer(params[1])
                print(response)
        else:
            print(answer)


if __name__ == '__main__':
    main()
