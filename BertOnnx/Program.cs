using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using MoreLinq.Extensions;

namespace BertOnnx
{
    class Program
    {
        private static readonly Stopwatch Stopwatch = new Stopwatch();

        static void Main(string[] args)
        {
            var settings = new Settings();

            Console.Write("Reading Hugging Face Config...");

            var config = HuggingFace.Config.FromFile(settings.ConfigPath);

            Console.WriteLine("Done");

            Console.Write("Reading Vocabulary...");

            var tokenizer = WordPieceTokenizer.FromVocabularyFile(settings.VocabPath);

            Console.WriteLine("Done");

            Console.WriteLine("Tokenizing...");

            var tokens = tokenizer.Tokenize(args[0]).ToArray();

            var padded = tokens.Select(t => (long)t.VocabularyIndex).Concat(Enumerable.Repeat(0L, settings.SequenceLength - tokens.Length)).ToArray();

            Console.WriteLine($"[{string.Join(',', tokens)}]");

            Console.WriteLine("...Done");

            var attentionMask = Enumerable.Repeat(1L, padded.Length).ToArray();

            var feature = new Feature { Tokens = padded, Attention = attentionMask };

            Console.Write("Creating Prediction Engine...");

            var loadingTime = Stopwatch.StartNew();
            var engine = Prediction.Engine<Feature, Result>.Create(settings, padded.Length);
            loadingTime.Stop();

            Console.WriteLine($"Done ({loadingTime.ElapsedMilliseconds} milliseconds)");

            Console.Write("Performing inference...");

            var inferendTime = Stopwatch.StartNew();
            var result = engine.Predict(feature);
            inferendTime.Stop();

            Console.WriteLine($"Done ({inferendTime.ElapsedMilliseconds} milliseconds)");

            tokens
                .Zip(result.Output.Batch(9).ToArray(), (token, values) => (Token: token, Values: values))
                .GroupBy(tuple => (WordIndex: tuple.Token.WordIndex, Word: tuple.Token.Word))
                .Select(group => GetWordCategory(config, group.Key.WordIndex, group.Key.Word, group.SelectMany(g => g.Values)))
                .Where(tuple => tuple.Category > 0)
                .ForEach(tuple => Console.WriteLine($"Word: {tuple.Word}, Label: {tuple.Label}, Score: {tuple.Score}"));
        }

        private static (int WordIndex, string Word, int Category, string Label, float Score) GetWordCategory(HuggingFace.Config config, int wordIndex, string word, IEnumerable<float> values)
        {
            return values
                .Select((v, i) => (Value: v, Index: i))
                .GroupBy(values => values.Index % 9)
                .Select((group, index) => (Category: index, Value: group.Average(g => g.Value)))
                .Where(tuple => tuple.Value > 0.1)
                .OrderByDescending(tuple => tuple.Value)
                .Select(tuple => (wordIndex, word, tuple.Category, config.id2label[tuple.Category.ToString()], tuple.Value))
                .FirstOrDefault();
        }
    }
}
