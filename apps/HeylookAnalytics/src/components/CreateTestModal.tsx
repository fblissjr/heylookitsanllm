// src/components/CreateTestModal.tsx
// Modal for creating new A/B tests

import React, { useState } from 'react';
import {
  View,
  ScrollView,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import {
  Modal,
  Portal,
  Text,
  TextInput,
  Button,
  IconButton,
  Chip,
  useTheme,
  List,
  FAB,
  Surface,
} from 'react-native-paper';
import { useMutation } from '@tanstack/react-query';

import { useApi } from '../providers/ApiProvider';

interface TestVariant {
  id: string;
  name: string;
  template: string;
  isControl: boolean;
}

interface CreateTestModalProps {
  visible: boolean;
  onDismiss: () => void;
  onCreated: () => void;
}

export default function CreateTestModal({
  visible,
  onDismiss,
  onCreated,
}: CreateTestModalProps) {
  const theme = useTheme();
  const api = useApi();
  
  const [testName, setTestName] = useState('');
  const [hypothesis, setHypothesis] = useState('');
  const [targetSampleSize, setTargetSampleSize] = useState('100');
  const [variants, setVariants] = useState<TestVariant[]>([
    { id: '1', name: 'Control', template: '', isControl: true },
    { id: '2', name: 'Variant A', template: '', isControl: false },
  ]);
  
  const createTest = useMutation({
    mutationFn: async () => {
      // Create test
      const testId = Date.now().toString();
      
      await api.execute(
        `INSERT INTO ab_tests (id, name, hypothesis, target_sample_size, status) 
         VALUES (?, ?, ?, ?, 'draft')`,
        [testId, testName, hypothesis, parseInt(targetSampleSize)]
      );
      
      // Create variants
      for (const variant of variants) {
        await api.execute(
          `INSERT INTO test_variants (id, test_id, name, prompt_template, is_control) 
           VALUES (?, ?, ?, ?, ?)`,
          [Date.now().toString(), testId, variant.name, variant.template, variant.isControl]
        );
      }
      
      return testId;
    },
    onSuccess: () => {
      onCreated();
      resetForm();
    },
  });
  
  const addVariant = () => {
    const letter = String.fromCharCode(65 + variants.filter(v => !v.isControl).length);
    setVariants([
      ...variants,
      {
        id: Date.now().toString(),
        name: `Variant ${letter}`,
        template: '',
        isControl: false,
      },
    ]);
  };
  
  const removeVariant = (id: string) => {
    setVariants(variants.filter(v => v.id !== id));
  };
  
  const updateVariant = (id: string, field: keyof TestVariant, value: any) => {
    setVariants(
      variants.map(v =>
        v.id === id ? { ...v, [field]: value } : v
      )
    );
  };
  
  const resetForm = () => {
    setTestName('');
    setHypothesis('');
    setTargetSampleSize('100');
    setVariants([
      { id: '1', name: 'Control', template: '', isControl: true },
      { id: '2', name: 'Variant A', template: '', isControl: false },
    ]);
  };
  
  const canCreate = testName && hypothesis && variants.every(v => v.template);
  
  return (
    <Portal>
      <Modal
        visible={visible}
        onDismiss={onDismiss}
        contentContainerStyle={styles.modal}
      >
        <KeyboardAvoidingView
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
          style={styles.container}
        >
          <ScrollView>
            <View style={styles.header}>
              <Text variant="headlineSmall">Create A/B Test</Text>
              <IconButton icon="close" onPress={onDismiss} />
            </View>
            
            <View style={styles.content}>
              {/* Test Details */}
              <TextInput
                label="Test Name"
                value={testName}
                onChangeText={setTestName}
                mode="outlined"
                style={styles.input}
              />
              
              <TextInput
                label="Hypothesis"
                value={hypothesis}
                onChangeText={setHypothesis}
                mode="outlined"
                multiline
                numberOfLines={3}
                style={styles.input}
              />
              
              <TextInput
                label="Target Sample Size (per variant)"
                value={targetSampleSize}
                onChangeText={setTargetSampleSize}
                mode="outlined"
                keyboardType="numeric"
                style={styles.input}
              />
              
              {/* Variants */}
              <View style={styles.variantsSection}>
                <View style={styles.sectionHeader}>
                  <Text variant="titleMedium">Prompt Variants</Text>
                  <Button
                    mode="text"
                    onPress={addVariant}
                    icon="plus"
                    compact
                  >
                    Add Variant
                  </Button>
                </View>
                
                {variants.map((variant, index) => (
                  <Surface key={variant.id} style={styles.variantCard} elevation={1}>
                    <View style={styles.variantHeader}>
                      <TextInput
                        value={variant.name}
                        onChangeText={(text) => updateVariant(variant.id, 'name', text)}
                        mode="flat"
                        dense
                        style={styles.variantName}
                      />
                      {variant.isControl && (
                        <Chip compact>Control</Chip>
                      )}
                      {!variant.isControl && (
                        <IconButton
                          icon="delete"
                          size={20}
                          onPress={() => removeVariant(variant.id)}
                        />
                      )}
                    </View>
                    
                    <TextInput
                      label="Prompt Template"
                      value={variant.template}
                      onChangeText={(text) => updateVariant(variant.id, 'template', text)}
                      mode="outlined"
                      multiline
                      numberOfLines={4}
                      style={styles.variantTemplate}
                      placeholder="Enter your prompt here. Use {variable} for dynamic content."
                    />
                  </Surface>
                ))}
              </View>
              
              {/* Quick Templates */}
              <View style={styles.templatesSection}>
                <Text variant="titleSmall" style={styles.templatesTitle}>
                  Quick Start Templates
                </Text>
                <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                  <Chip
                    onPress={() => applyTemplate('tone')}
                    style={styles.templateChip}
                  >
                    Tone Comparison
                  </Chip>
                  <Chip
                    onPress={() => applyTemplate('length')}
                    style={styles.templateChip}
                  >
                    Length Test
                  </Chip>
                  <Chip
                    onPress={() => applyTemplate('structure')}
                    style={styles.templateChip}
                  >
                    Structure Test
                  </Chip>
                </ScrollView>
              </View>
            </View>
            
            {/* Actions */}
            <View style={styles.actions}>
              <Button
                mode="outlined"
                onPress={onDismiss}
                style={styles.actionButton}
              >
                Cancel
              </Button>
              <Button
                mode="contained"
                onPress={() => createTest.mutate()}
                loading={createTest.isPending}
                disabled={!canCreate}
                style={styles.actionButton}
              >
                Create Test
              </Button>
            </View>
          </ScrollView>
        </KeyboardAvoidingView>
      </Modal>
    </Portal>
  );
  
  function applyTemplate(type: string) {
    switch (type) {
      case 'tone':
        setTestName('Tone Style Comparison');
        setHypothesis('A more casual tone will improve user engagement');
        setVariants([
          {
            id: '1',
            name: 'Formal',
            template: 'Please provide a detailed analysis of {topic}, including comprehensive insights and implications.',
            isControl: true,
          },
          {
            id: '2',
            name: 'Casual',
            template: 'Hey! Tell me about {topic} - what are the key things I should know?',
            isControl: false,
          },
        ]);
        break;
      
      case 'length':
        setTestName('Response Length Test');
        setHypothesis('Users prefer concise responses for quick tasks');
        setVariants([
          {
            id: '1',
            name: 'Detailed',
            template: 'Explain {concept} in detail with examples and context.',
            isControl: true,
          },
          {
            id: '2',
            name: 'Concise',
            template: 'Briefly explain {concept} in 2-3 sentences.',
            isControl: false,
          },
        ]);
        break;
      
      case 'structure':
        setTestName('Output Structure Test');
        setHypothesis('Structured outputs are easier to understand');
        setVariants([
          {
            id: '1',
            name: 'Paragraph',
            template: 'Describe the steps to {task}.',
            isControl: true,
          },
          {
            id: '2',
            name: 'Numbered List',
            template: 'List the steps to {task} as a numbered list.',
            isControl: false,
          },
        ]);
        break;
    }
  }
}

const styles = StyleSheet.create({
  modal: {
    backgroundColor: 'white',
    margin: 20,
    maxHeight: '90%',
    borderRadius: 8,
  },
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  content: {
    padding: 16,
  },
  input: {
    marginBottom: 16,
  },
  variantsSection: {
    marginTop: 24,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  variantCard: {
    padding: 12,
    marginBottom: 12,
    borderRadius: 8,
  },
  variantHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  variantName: {
    flex: 1,
    marginRight: 8,
  },
  variantTemplate: {
    fontSize: 14,
  },
  templatesSection: {
    marginTop: 24,
  },
  templatesTitle: {
    marginBottom: 8,
  },
  templateChip: {
    marginRight: 8,
  },
  actions: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    padding: 16,
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
    gap: 12,
  },
  actionButton: {
    minWidth: 100,
  },
});