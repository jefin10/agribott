import 'package:flutter/material.dart';

class Screen extends StatelessWidget{
  const Screen({super.key, required this.page});
  final Widget page;  
  @override
  Widget build(BuildContext context) {
      return page; 
  }
}