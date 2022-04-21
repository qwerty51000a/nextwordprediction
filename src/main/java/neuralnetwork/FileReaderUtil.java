package neuralnetwork;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

/**
 * A class to read lines (sentences) from a file. 
 * 
 * @author Ibrahim
 *
 */
public class FileReaderUtil {

	String path;

	public FileReaderUtil(String path) {
		this.path = path;
	}

	/**
	 * 
	 * 
	 * Reads sentences from a file. Each sentence words are separated by
	 * wordDelimtier which is usually a space. Each sentence is assumed to be on a
	 * separate line. The first numberOfSkippedLines lines will be skipped. 
	 * 
	 * @param wordDelimiter delimitter that separates between a word in a sentence
	 * @param numberOfSkippedLines first number of lines to be skipped
	 * @return
	 */
	public List<List<String>> readSentences(String wordDelimiter, int numberOfSkippedLines) {

		BufferedReader br = null;
		List<List<String>> listOfSentences = new ArrayList<List<String>>();

		try {

			br = new BufferedReader(new FileReader(path));

			String line;

			while ((line = br.readLine()) != null) {

				if (numberOfSkippedLines-- > 0)
					continue;

				List<String> sentence = new ArrayList<>();

				String[] tokens = line.split(wordDelimiter);

				for (String token : tokens)
					sentence.add(token);

				listOfSentences.add(sentence);

			}
		} catch (Exception e) {
			System.out.println("Exception while trying to read file " + path);
			e.printStackTrace();

		} finally {
			try {
				br.close();
			} catch (Exception e2) {
				System.out.println("Exception while trying to close BufferedReader br while reading file " + path);
				e2.printStackTrace();
			}

		}

		return listOfSentences;
	}

}
