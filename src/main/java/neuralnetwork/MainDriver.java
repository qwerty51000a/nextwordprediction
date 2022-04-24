package neuralnetwork;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.stream.Collectors;

/**
 * This is the main method of the program "Next Word Prediction". This program
 * uses LSTM (Long Short-Term Memory) recurrent neural network to predict the
 * next word in a sentence. The program trains the LSTM with the sentences that
 * are stored in the file filePath. A sliding window will read each sentence 5
 * words at a time. The first four words will be used as input and the last word
 * will be used as an output for training the LSTM. One-hot encoding is used to
 * map words to vectors. A sentence shorter than 4 words will be ignored. The
 * program gives as output the next most probable word as predicted by the LSTM.
 * The four words in the List inputWords are used to predict the next word.
 * 
 * @author Ibrahim
 *
 */
public class MainDriver {

	public static void main(String[] args) {

		System.out.println("Welcome to the next word prediction program  \n");
		
		// The program will predict the next word of these four words. (Note: They have
		// to be four words)
		List<String> inputWords = List.of("Fall", "seven", "times", "stand");

		String filePath = "sampleText.txt";

		FileReaderUtil fru = new FileReaderUtil(filePath);

		List<List<String>> listOfSentences = fru.readSentences(" ", 2);

		List<String> listOfAllWords = listOfSentences.stream().flatMap(Collection::stream).collect(Collectors.toList());

		System.out.println("total number of words: " + listOfAllWords.stream().count());

		TextToSequenceTokenizer textToSequenceTokenizer = new TextToSequenceTokenizer();

		final LinkedHashMap<String, Matrix> mapOfUniqueWords = textToSequenceTokenizer
				.mapFromWordsToMatrices(listOfAllWords);

		final List<String> listOfUniqueWords = textToSequenceTokenizer.getListOfUniqueWords(mapOfUniqueWords);

		System.out.println("number of unique words " + mapOfUniqueWords.size());

		LSTM lstm = new LSTM(mapOfUniqueWords.size(), mapOfUniqueWords.size(), 0.1);

		List<List<Matrix>> inputMatrices = new ArrayList<List<Matrix>>();

		List<Matrix> outputMatrices = new ArrayList<Matrix>();

		for (final List<String> sentence : listOfSentences) {
			List<Matrix> sentenceAsMatrices = textToSequenceTokenizer.convertFromWordsToMatrices(sentence,
					mapOfUniqueWords);

			// Only consider sentences of length 5 or more
			if (sentenceAsMatrices.size() < 5)
				continue;

			System.out.println("sentence.size: " + sentence.size());

			for (int i = 0; i < sentenceAsMatrices.size() - 4; i++) {
				List<Matrix> fourWords = new ArrayList<Matrix>();

				for (int j = i; j < i + 4; j++) {

					fourWords.add(sentenceAsMatrices.get(j));
				}

				inputMatrices.add(fourWords);

				outputMatrices.add(sentenceAsMatrices.get(i + 4));
			}

		}

		lstm.train(inputMatrices, outputMatrices, 500, -3, 3);

		// testing

		System.out.println(inputWords);

		List<Matrix> inputWordsAsMatrices = textToSequenceTokenizer.convertFromWordsToMatrices(inputWords,
				mapOfUniqueWords);
		Matrix nextWordMatrix = lstm.getOutput(inputWordsAsMatrices);

		String nextWord = textToSequenceTokenizer.getWordFromOutputMatrix(nextWordMatrix, listOfUniqueWords);

		System.out.println("The next word should be    " + nextWord);

	}

}
