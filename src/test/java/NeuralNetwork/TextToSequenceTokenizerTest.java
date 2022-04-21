package NeuralNetwork;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.Before;
import org.junit.Test;

import neuralnetwork.Matrix;
import neuralnetwork.TextToSequenceTokenizer;

import static org.junit.Assert.assertEquals;

public class TextToSequenceTokenizerTest {

	TextToSequenceTokenizer textToSequenceTokenizer = new TextToSequenceTokenizer();

	List<String> list1;

	List<String> list2;

	List<String> list3;

	LinkedHashMap<String, Matrix> mapOfUniqueWords1;

	LinkedHashMap<String, Matrix> mapOfUniqueWords2;

	LinkedHashMap<String, Matrix> mapOfUniqueWords3;

	List<String> listOfUniqueWords1;

	List<String> listOfUniqueWords2;

	List<String> listOfUniqueWords3;

	@Before
	public void before() {

		list1 = List.of("a", "a", "a", "o", "o", "b", "w");

		list2 = List.of("a", "a", "b", "o", "o", "o", "o", "o");

		list3 = List.of("a", "a", "a", "a", "a");

		mapOfUniqueWords1 = textToSequenceTokenizer.mapFromWordsToMatrices(list1);

		mapOfUniqueWords2 = textToSequenceTokenizer.mapFromWordsToMatrices(list2);

		mapOfUniqueWords3 = textToSequenceTokenizer.mapFromWordsToMatrices(list3);

		listOfUniqueWords1 = textToSequenceTokenizer.getListOfUniqueWords(mapOfUniqueWords1);

		listOfUniqueWords2 = textToSequenceTokenizer.getListOfUniqueWords(mapOfUniqueWords2);

		listOfUniqueWords3 = textToSequenceTokenizer.getListOfUniqueWords(mapOfUniqueWords3);

	}

	@Test
	public void test1() {

		// Check counts of each character

		Map<String, Integer> map1 = textToSequenceTokenizer.getWordsToIntegerMapping(list1);

		assertEquals(map1.get("o"), Integer.valueOf(2));

		assertEquals(map1.get("a"), Integer.valueOf(3));

		assertEquals(map1.get("b"), Integer.valueOf(1));

		assertEquals(map1.get("w"), Integer.valueOf(1));

		assertEquals(map1.get("p"), null);

		assertEquals(map1.get("q"), null);

		Map<String, Integer> map2 = textToSequenceTokenizer.getWordsToIntegerMapping(list2);

		assertEquals(map2.get("o"), Integer.valueOf(5));

		assertEquals(map2.get("a"), Integer.valueOf(2));

		assertEquals(map2.get("b"), Integer.valueOf(1));

		assertEquals(map2.get("p"), null);

		assertEquals(map2.get("q"), null);

		Map<String, Integer> map3 = textToSequenceTokenizer.getWordsToIntegerMapping(list3);

		assertEquals(map3.get("a"), Integer.valueOf(5));

		assertEquals(map3.get("b"), null);

		assertEquals(map3.get("a"), Integer.valueOf(5));

		assertEquals(map3.get("p"), null);

		assertEquals(map3.get("q"), null);

		Map<String, Integer> map4 = textToSequenceTokenizer.getWordToIntegerMappingSorted(list1);

		assertEquals(map4.get("o"), Integer.valueOf(1));

		Map<String, Integer> map5 = textToSequenceTokenizer.getWordToIntegerMappingSorted(list2);

		assertEquals(map5.get("o"), Integer.valueOf(0));

		assertEquals(map5.get("a"), Integer.valueOf(1));

		assertEquals(map5.get("b"), Integer.valueOf(2));

		Map<String, Integer> map6 = textToSequenceTokenizer.getWordToIntegerMappingSorted(list3);

		assertEquals(map6.get("a"), Integer.valueOf(0));

	}

	@Test
	// testing one hot encoding
	public void testMapFromWordsToMatrices() {

		testMapFromWordsToMatricesHelper(mapOfUniqueWords1);

		testMapFromWordsToMatricesHelper(mapOfUniqueWords2);

		testMapFromWordsToMatricesHelper(mapOfUniqueWords3);

	}

	public void testMapFromWordsToMatricesHelper(final LinkedHashMap<String, Matrix> mapOfUniqueWords) {
		Set<String> set = new HashSet<String>();

		int indexThatIsOne = 0;

		// LinkedHashMap method keySet() returns the keys in the order of initial
		// insertion
		for (String key : mapOfUniqueWords.keySet()) {
			Matrix output = mapOfUniqueWords.get(key);

			for (int j = 0; j < output.getRows(); j++) {
				if (indexThatIsOne == j)
					assert (output.getValue(j, 0) == 1);
				else
					assert (output.getValue(j, 0) == 0);
			}

			indexThatIsOne++;

		}

		set.clear();

		for (String key : mapOfUniqueWords.keySet()) {
			set.add(key);
		}

		for (String key : set) {

			assert (mapOfUniqueWords.containsKey(key));
		}

		assert (mapOfUniqueWords.size() == set.size());

	}

	@Test
	public void testGetListOfUniqueWords() {

		assert (listOfUniqueWords1.size() == 4);

		assert (listOfUniqueWords2.size() == 3);

		assert (listOfUniqueWords3.size() == 1);

		assert (listOfUniqueWords1.get(0).equals("a"));

		assert (listOfUniqueWords1.get(1).equals("o"));

		assert (listOfUniqueWords1.get(2).equals("b"));

		assert (listOfUniqueWords1.get(3).equals("w"));

		assert (listOfUniqueWords2.get(0).equals("a"));

		assert (listOfUniqueWords2.get(1).equals("b"));

		assert (listOfUniqueWords2.get(2).equals("o"));

		assert (listOfUniqueWords3.get(0).equals("a"));

	}

	

	@Test
	public void testGetWordFromOutputMatrix() {

		testGetWordFromOutputMatrixHelper(listOfUniqueWords1);

		testGetWordFromOutputMatrixHelper(listOfUniqueWords2);

		testGetWordFromOutputMatrixHelper(listOfUniqueWords3);

	}

	public void testGetWordFromOutputMatrixHelper(final List<String> listOfUniqueWords) {

		Matrix[] outputMatrices = new Matrix[listOfUniqueWords.size()];

		for (int i = 0; i < listOfUniqueWords.size(); i++) {

			outputMatrices[i] = new Matrix.Builder().setRows(listOfUniqueWords.size()).setColumns(1).build();

			outputMatrices[i].setValue(i, 0, 1);

		}

		String[] words = new String[listOfUniqueWords.size()];

		for (int i = 0; i < listOfUniqueWords.size(); i++) {

			words[i] = textToSequenceTokenizer.getWordFromOutputMatrix(outputMatrices[i], listOfUniqueWords);

		}

		String[] expectedWords = new String[listOfUniqueWords.size()];

		for (int i = 0; i < listOfUniqueWords.size(); i++) {

			expectedWords[i] = listOfUniqueWords.get(i);

		}

		for (int i = 0; i < listOfUniqueWords.size(); i++) {
			assert (words[i].equals(expectedWords[i]));
		}

	}

}
