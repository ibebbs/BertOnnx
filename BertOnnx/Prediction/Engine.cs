using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Text;

namespace BertOnnx.Prediction
{
    public class Engine<TFeature, TResult>
        where TFeature : class
        where TResult : class, new()
    {
        private static ITransformer CreateModel(MLContext context, Settings configuration, int tokens)
        {
            bool hasGpu = false;

            var dataView = context.Data
                .LoadFromEnumerable(new List<TFeature>());

            var pipeline = context.Transforms
                            .ApplyOnnxModel(
                                modelFile: configuration.ModelPath,
                                shapeDictionary: new Dictionary<string, int[]>
                                {
                                    { "input_ids", new [] { 1, configuration.SequenceLength } },
                                    { "attention_mask", new [] { 1, configuration.SequenceLength } }
                                },
                                inputColumnNames: configuration.ModelInput,
                                outputColumnNames: configuration.ModelOutput, 
                                gpuDeviceId: hasGpu ? 0 : (int?)null);

            var transformer = pipeline.Fit(dataView);

            return transformer;
        }

        public static Engine<TFeature, TResult> Create(Settings configuration, int tokens)
        {
            var context = new MLContext();

            ITransformer transformer = CreateModel(context, configuration, tokens);

            var engine = context.Model.CreatePredictionEngine<TFeature, TResult>(transformer);

            return new Engine<TFeature, TResult>(engine);
        }

        private PredictionEngine<TFeature, TResult> _engine;

        public Engine(PredictionEngine<TFeature, TResult> engine)
        {
            _engine = engine;
        }

        public TResult Predict(TFeature feature)
        {
            var result = _engine.Predict(feature);

            return result;
        }
    }
}
