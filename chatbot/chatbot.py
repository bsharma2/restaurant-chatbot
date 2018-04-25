from chatbot_classifier import ChatbotClassifier


while True:
	sentence = input(">> ")
	chatbot = ChatbotClassifier()
	chatbot.set_sentence(sentence)
	print(chatbot.classify())
