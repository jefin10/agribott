import 'package:flutter/material.dart';
import 'package:mobile_app/home.dart';

void main() {
  runApp(const AgriBott());
}

class AgriBott extends StatelessWidget {
  const AgriBott({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        body: Home(),
      ),
    );
  
  }
}
