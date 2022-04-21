package neuralnetwork;

import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;
import java.util.List;
import java.util.ArrayList;
import java.util.Set;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.HashMap;

/**
 * A class for conversion between words and matrices.
 * 
 * @author Ibrahim
 *
 */
public class TextToSequenceTokenizer {

	public TextToSequenceTokenizer() {

	}

	/**
	 * Returns a map that maps from a word to its frequency of occurrence
	 * 
	 * @param listOfWords
	 * @return
	 */

	public Map<String, Integer> getWordsToIntegerMapping(List<String> listOfWords) {
		Map<String, Integer> map = new HashMap<String, Integer>();

		for (String word : listOfWords) {
			word = word.strip();
			map.put(word, map.getOrDefault(word, 0) + 1);

		}

		return map;
	}

	/**
	 * Returns a map that maps from a word to the integers 0,1,2 with the words
	 * sorted by frequency of occurrence descending (E.g. word "a" -> 3 occurrences
	 * word "b" -> 5 occurrences word "u" -> 1 occurrence would return ["b", "a",
	 * "u"]
	 * 
	 * @param listOfWords
	 * @return
	 */
	public Map<String, Integer> getWordToIntegerMappingSorted(List<String> listOfWords) {
		Map<String, Integer> unsortedCountingMap = getWordsToIntegerMapping(listOfWords);

		List<String> list = unsortedCountingMap.entrySet().stream()
				.sorted(Entry.<String, Integer>comparingByValue().reversed()).map(Entry::getKey)
				.collect(Collectors.toList());

		Map<String, Integer> map = new HashMap<String, Integer>();

		for (int i = 0; i < list.size(); i++) {
			map.put(list.get(i), i);
		}

		return map;
	}

	/**
	 * 
	 * Return a map that maps from words to matrices using one hot encoding
	 * 
	 * @param listOfWords
	 * @return
	 */
	public LinkedHashMap<String, Matrix> mapFromWordsToMatrices(List<String> listOfAllWords) {
		LinkedHashMap<String, Matrix> mapOfUniqueWords = new LinkedHashMap<>();

		Set<String> setOfUniqueWords = new LinkedHashSet<String>();

		for (String word : listOfAllWords)
			setOfUniqueWords.add(word);

		int numberOfUniqueWords = setOfUniqueWords.size();

		int i = 0;

		for (String word : setOfUniqueWords) {
			Matrix mat = new Matrix.Builder().setRows(numberOfUniqueWords).setColumns(1).build();

			// set one digit per matrix
			mat.setValue(i, 0, 1);
			i++;

			mapOfUniqueWords.put(word, mat);

		}

		return mapOfUniqueWords;
	}

	/**
	 * Returns a list of words in a sorted manner. Remember that keySet() method in
	 * LinkedHashMap returns a set that is SORTED.
	 * 
	 * @param mapOfWords
	 * @return
	 */
	public List<String> getListOfUniqueWords(final LinkedHashMap<String, Matrix> mapOfUniqueWords) {
		List<String> listOfUniqueWords = new ArrayList<>();

		for (String s : mapOfUniqueWords.keySet()) {
			listOfUniqueWords.add(s);
		}
		return listOfUniqueWords;

	}

	/**
	 * Get the word from the listOfUniqueWords (a list is sorted in the order of
	 * inserting so get(int index) could be used. One hot encoding is assumed so
	 * therefore the row with the maximum value is assumed to be 1.
	 * 
	 * @param outputMatrix      the output matrix that is the output of the lstm
	 * @param listOfUniqueWords the list of the unique words.
	 * @return
	 */
	public String getWordFromOutputMatrix(final Matrix outputMatrix, final List<String> listOfUniqueWords) {

		int index = 0;

		double max = -Double.MAX_VALUE; // most negative double value in java

		for (int i = 0; i < outputMatrix.getRows(); i++) {
			if (max < outputMatrix.getValue(i, 0)) {
				index = i;
				max = outputMatrix.getValue(i, 0);
			}

		}

		return listOfUniqueWords.get(index);
	}

	/**
	 * Converts the list of words to a list of matrices using the Map mapOfWords.
	 * 
	 * @param listOfWords List of words that will be converted to a list of matrices
	 * @param mapOfWords  Map of words that is used for the conversion between words
	 *                    to matrices
	 * @return
	 */
	public List<Matrix> convertFromWordsToMatrices(final List<String> listOfWords,
			final Map<String, Matrix> mapOfWords) {
		List<Matrix> listOfMatrices = new ArrayList<Matrix>();

		for (String word : listOfWords) {

			if (!mapOfWords.containsKey(word))
				throw new IllegalStateException(" A word does not exist in the mapOfWords provieded");

			Matrix mat = mapOfWords.get(word);

			listOfMatrices.add(mat);

		}

		return listOfMatrices;
	}

}
